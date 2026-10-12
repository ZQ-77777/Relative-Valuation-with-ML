"""Synthetic data with the same structure as the raw WRDS tables, to run the whole pipeline before real data are available.
Values are random and carry no economic meaning; they only test the code."""
import numpy as np
import pandas as pd

SIC_TEXT = """ 1 Agric  Agriculture
          0100-0199 Agricultural production - crops
 13 Drugs  Pharmaceutical Products
          2830-2836 Drugs
 36 Softw  Computer Software
          7370-7372 Services - computer programming
 45 Banks  Banking
          6000-6099 Banking
 49 Other  Almost Nothing
          4950-4959 Sanitary services
"""


def make_raw(n_firms=300, start="1974-01-31", end="1982-12-31", seed=0):
    rng = np.random.default_rng(seed)
    months = pd.date_range(start, end, freq="ME")
    quarters = months[months.month.isin([3, 6, 9, 12])]
    firms = pd.DataFrame({"permno": np.arange(10001, 10001 + n_firms),
                          "gvkey": [f"{i:06d}" for i in range(1, n_firms + 1)],
                          "size": rng.lognormal(6, 1.2, n_firms),
                          "quality": rng.normal(0, 1, n_firms),
                          "siccd": rng.choice([150, 2834, 7372, 6020, 4955], n_firms)})
    # ---- Compustat fundq ----
    rows = []
    for f in firms.itertuples():
        at = f.size
        for d in quarters:
            at *= np.exp(rng.normal(0.01, 0.03))
            lt = at * rng.uniform(0.3, 0.7)
            sale = at * rng.uniform(0.15, 0.35)
            cogs, xsga, dp = sale * 0.6, sale * (0.2 - 0.03 * f.quality), at * 0.01
            oiadp = sale - cogs - xsga - dp
            xint = lt * 0.01
            pi = oiadp - xint
            txt = 0.3 * max(pi, 0)
            ib = pi - txt
            rows.append(dict(gvkey=f.gvkey, datadate=d, rdq=d + pd.Timedelta(days=40),
                             fyearq=d.year, fqtr=(d.month // 3), atq=at, ltq=lt,
                             ceqq=at - lt, seqq=at - lt, pstkq=0.0, txditcq=at * 0.01,
                             cheq=at * rng.uniform(0.05, 0.3), actq=at * 0.45,
                             lctq=lt * 0.4, ppentq=at * 0.35, invtq=at * 0.1,
                             rectq=at * 0.12, apq=at * 0.08, dlttq=lt * 0.4,
                             dlcq=lt * 0.1, icaptq=at - lt * 0.5, mibq=0.0,
                             saleq=sale, cogsq=cogs, xsgaq=xsga, xrdq=sale * 0.05 * (f.siccd == 2834),
                             dpq=dp, xintq=xint, oiadpq=oiadp, oibdpq=oiadp + dp, piq=pi,
                             txtq=txt, ibq=ib, ibcomq=ib, ibadjq=ib, nopiq=0.0, spiq=0.0,
                             miiq=0.0, capx_q=at * 0.02, oancf_q=ib + dp, dv_q=max(ib, 0) * 0.3))
    fq = pd.DataFrame(rows)
    for c in ["capx", "oancf", "dv"]:                        # convert to year-to-date (YTD)
        fq[c + "y"] = fq.groupby(["gvkey", "fyearq"])[c + "_q"].cumsum()
    fq = fq.drop(columns=["capx_q", "oancf_q", "dv_q"])
    # ---- CRSP msf ----
    q_last = fq.set_index(["gvkey", "datadate"])
    m_rows = []
    for f in firms.itertuples():
        g = fq[fq.gvkey == f.gvkey].set_index("datadate")
        for d in months:
            past = g[g.index <= d]
            be = past["ceqq"].iloc[-1] if len(past) else f.size * 0.5
            me = be * np.exp(0.5 + 0.6 * f.quality + rng.normal(0, 0.3))
            shrout = 10_000.0
            m_rows.append(dict(permno=f.permno, date=d - pd.Timedelta(days=rng.integers(0, 3)),
                               prc=-me / shrout * 1000 if rng.random() < 0.1 else me / shrout * 1000,
                               shrout=shrout, ret=rng.normal(0.01, 0.08), retx=np.nan))
    msf = pd.DataFrame(m_rows)
    names = pd.DataFrame({"permno": firms.permno, "namedt": pd.Timestamp("1960-01-01"),
                          "nameendt": pd.NaT, "shrcd": 10,
                          "exchcd": rng.choice([1, 3], n_firms), "siccd": firms.siccd,
                          "ticker": "X", "comnam": [f"FIRM {i}" for i in range(n_firms)]})
    delist = pd.DataFrame({"permno": [10001], "dlstdt": [pd.Timestamp(end)], "dlret": [-0.3]})
    link = pd.DataFrame({"gvkey": firms.gvkey, "permno": firms.permno,
                         "linktype": "LU", "linkprim": "P",
                         "linkdt": pd.Timestamp("1960-01-01"), "linkenddt": pd.NaT})
    ff = pd.DataFrame({"date": months})
    for c in ["mktrf", "smb", "hml", "rmw", "cma", "umd"]:
        ff[c] = rng.normal(0.005, 0.04, len(months))
    ff["rf"] = 0.004
    return msf, names, delist, link, fq, ff
