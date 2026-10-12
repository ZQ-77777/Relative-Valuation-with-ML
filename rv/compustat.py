"""Step 3: Compustat quarterly data -> one row per firm-quarter of base accounting variables.

Key steps:
  1. convert year-to-date items (capxy, oancfy, dvy) to single quarters;
  2. sum flow items over the last four quarters (TTM; paper: "aggregating quarter t-3 to t");
  3. compute book equity be (Davis, Fama and French 2000);
  4. create same-quarter-prior-year lags l_x and changes d_x;
  5. availability date avail = fiscal period end + 3 months.
Steps 3-5 live in finalize_quarterly, which the free-data version (rv/free/) reuses.
"""
import numpy as np
import pandas as pd
from .config import INFO_LAG_MONTHS

STOCK = ["atq", "ltq", "ceqq", "seqq", "pstkq", "txditcq", "cheq", "actq",
         "lctq", "ppentq", "invtq", "rectq", "apq", "dlttq", "dlcq", "icaptq", "mibq"]
FLOW = ["saleq", "cogsq", "xsgaq", "xrdq", "dpq", "xintq", "oiadpq", "oibdpq",
        "piq", "txtq", "ibq", "ibcomq", "ibadjq", "nopiq", "spiq", "miiq"]
YTD = ["capxy", "oancfy", "dvy"]

# variables that need lags l_ and changes d_ (from the formulas in Table A.1)
LAGGED = ["at", "be", "ceq", "lt", "act", "lct", "che", "dlc", "dltt", "pstk",
          "invt", "rect", "ap", "sale", "cogs", "xsga", "ib", "dp", "ppent",
          "capx", "icapt", "txditc", "mib"]


def _strip_q(c: str) -> str:
    """'atq' -> 'at'; 'capxq' -> 'capx'."""
    return c[:-1]


def build_comp_quarterly(fundq: pd.DataFrame) -> pd.DataFrame:
    """
    Input   raw comp.fundq: gvkey, datadate, fyearq, fqtr and the STOCK/FLOW/YTD columns
            ($ millions)
    Output  one row per gvkey-datadate:
            base variables (at, lt, ceq, be, sale, cogs, ib, ebit, ebitda, ocf, capx ...),
            lags l_*, changes d_*, availability date avail
    """
    q = fundq.copy()
    q["datadate"] = pd.to_datetime(q["datadate"])
    q = (q.dropna(subset=["fyearq", "fqtr"])
           .sort_values(["gvkey", "datadate"])
           .drop_duplicates(["gvkey", "datadate"], keep="last")
           .reset_index(drop=True))

    # 1) YTD -> quarter: Q1 as reported; later quarters = this YTD - previous YTD
    for c in YTD:
        prev = q.groupby(["gvkey", "fyearq"])[c].shift(1)
        q[c[:-1] + "q"] = np.where(q["fqtr"] == 1, q[c], q[c] - prev)

    g = q.groupby("gvkey")
    # 2) TTM: sum of the last four quarters, which must be consecutive (about nine months apart)
    span = (q["datadate"] - g["datadate"].shift(3)).dt.days
    consecutive = span.between(250, 300)
    for c in FLOW + [y[:-1] + "q" for y in YTD]:
        s = g[c].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
        q[_strip_q(c)] = s.where(consecutive)

    # stock (balance sheet) items: value at quarter end
    for c in STOCK:
        q[_strip_q(c)] = q[c]

    q["avail"] = q["datadate"] + pd.offsets.MonthEnd(INFO_LAG_MONTHS)
    return finalize_quarterly(q)


def finalize_quarterly(q: pd.DataFrame) -> pd.DataFrame:
    """
    Shared final step (used by both the WRDS and the free-data versions).
    Input   one row per gvkey-datadate with base variables (at, lt, ceq, seq, pstk, txditc, sale, oiadp,
            oibdp, oancf ...) and availability date avail
    Output  adds be, the ebit/ebitda/ocf aliases, lags l_* and changes d_*; keeps only needed columns
    """
    q = q.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    for c in [_strip_q(x) for x in STOCK + FLOW] + [y[:-1] for y in YTD]:
        if c not in q:
            q[c] = np.nan                      # items absent from free data stay missing; features handle NaN

    # 3) book equity: stockholders' equity + deferred taxes - preferred stock
    seq = q["seq"].fillna(q["ceq"] + q["pstk"].fillna(0)).fillna(q["at"] - q["lt"])
    q["be"] = seq + q["txditc"].fillna(0) - q["pstk"].fillna(0)

    # aliases matching the notation in Table A.1
    q["ebit"] = q["oiadp"]
    q["ebitda"] = q["oibdp"]
    q["ocf"] = q["oancf"]

    # 4) four-quarter lag (same quarter last year), checking the gap is about one year
    g = q.groupby("gvkey")
    lag_ok = (q["datadate"] - g["datadate"].shift(4)).dt.days.between(350, 380)
    for c in LAGGED:
        q["l_" + c] = g[c].shift(4).where(lag_ok)
        q["d_" + c] = q[c] - q["l_" + c]

    keep = (["gvkey", "datadate", "avail", "fyearq", "fqtr", "seq", "be", "ebit",
             "ebitda", "ocf"]
            + [_strip_q(c) for c in STOCK + FLOW]
            + [y[:-1] for y in YTD]
            + [p + c for c in LAGGED for p in ("l_", "d_")])
    keep = list(dict.fromkeys(keep))   # de-duplicate, keep order
    return q[keep]
