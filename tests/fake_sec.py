"""Synthetic data with the structure of SEC companyfacts JSON, plus daily prices in yfinance format.
Includes the awkward cases of real data: comparatives repeated in later filings, restatements (10-K/A),
tag changes (RevenueFromContractWithCustomer... from 2018), stock splits, and YTD-only cash flows."""
import numpy as np
import pandas as pd

INC = {"Revenues": 1.0, "CostOfRevenue": 0.6, "SellingGeneralAndAdministrativeExpense": 0.2,
       "ResearchAndDevelopmentExpense": 0.04, "InterestExpense": 0.01, "OperatingIncomeLoss": 0.15,
       "IncomeTaxExpenseBenefit": 0.03, "NetIncomeLoss": 0.11}
CF = {"NetCashProvidedByUsedInOperatingActivities": 0.14, "PaymentsToAcquirePropertyPlantAndEquipment": 0.05,
      "DepreciationDepletionAndAmortization": 0.04, "PaymentsOfDividends": 0.02}
BS = {"Assets": 1.0, "Liabilities": 0.5, "AssetsCurrent": 0.45, "LiabilitiesCurrent": 0.2,
      "CashAndCashEquivalentsAtCarryingValue": 0.12, "StockholdersEquity": 0.5,
      "PropertyPlantAndEquipmentNet": 0.35, "InventoryNet": 0.1, "AccountsReceivableNetCurrent": 0.12,
      "AccountsPayableCurrent": 0.07, "LongTermDebtNoncurrent": 0.2, "LongTermDebtCurrent": 0.05}
QSTART = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}


def _qend(y, q):
    return pd.Timestamp(y, 3 * q, 1) + pd.offsets.MonthEnd(0)


def make_sec(n_firms=40, years=range(2008, 2021), seed=1):
    rng = np.random.default_rng(seed)
    docs, truth, daily, tick = [], [], [], []
    for i in range(n_firms):
        cik = 1000 + i
        size = rng.lognormal(7, 1)
        quality = rng.normal()
        switch = (i % 3 == 0)                    # a third of firms switch revenue tags in 2018
        split_date = pd.Timestamp("2015-06-15") if i == 0 else None
        shares0 = 1e8
        facts = {}

        def add(tag, unit, start, end, val, filed, form, fy, fp, accn):
            if unit == "USD":
                val = val * 1e6                         # as in real XBRL: amounts in dollars
            facts.setdefault(tag, {}).setdefault(unit, []).append(
                dict(start=None if start is None else start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), val=float(val), filed=filed.strftime("%Y-%m-%d"),
                     form=form, fy=fy, fp=fp, accn=accn))
        qsale = {}
        for y in years:
            for q in range(1, 5):
                qsale[(y, q)] = size * 0.25 * np.exp(0.02 * (y - 2008) + rng.normal(0, 0.05))
        bs_scale = {k: size * np.exp(0.02 * (k[0] - 2008)) for k in qsale}
        for y in years:
            for q in range(1, 5):
                E = _qend(y, q)
                form, fp = ("10-K", "FY") if q == 4 else ("10-Q", f"Q{q}")
                filed = E + pd.Timedelta(days=60 if q == 4 else 40)
                accn = f"{cik}-{y}{q}"
                rev_tag = "RevenueFromContractWithCustomerExcludingAssessedTax" if (switch and y >= 2018) else "Revenues"
                for yy in (y, y - 1):                       # current period + prior-year comparative
                    if (yy, q) not in qsale:
                        continue
                    Ey = _qend(yy, q)
                    # income statement: 10-Q reports 3-month and YTD values, 10-K the full year
                    periods = [(pd.Timestamp(yy, 1, 1), Ey)]
                    if q < 4:
                        m0, d0 = QSTART[q]
                        periods.append((pd.Timestamp(yy, m0, d0), Ey))
                    for tag, share in INC.items():
                        t = rev_tag if tag == "Revenues" else tag
                        if i == 3 and tag == "OperatingIncomeLoss":
                            continue                        # a firm that does not report operating income
                        if i == 2 and tag == "SellingGeneralAndAdministrativeExpense":
                            for t2, w in (("GeneralAndAdministrativeExpense", 0.7), ("SellingAndMarketingExpense", 0.3)):
                                for s, e in periods:
                                    qs = [k for k in qsale if k[0] == yy and _qend(*k) <= e and _qend(*k) > s]
                                    add(t2, "USD", s, e, w * share * sum(qsale[k] for k in qs), filed, form, y, fp, accn)
                            continue
                        for s, e in periods:
                            qs = [k for k in qsale if k[0] == yy and _qend(*k) <= e and _qend(*k) > s]
                            add(t, "USD", s, e, share * sum(qsale[k] for k in qs) * (1 + 0.1 * quality * (tag == "NetIncomeLoss")),
                                filed, form, y, fp, accn)
                    for tag, share in CF.items():           # cash flows: YTD only
                        qs = [k for k in qsale if k[0] == yy and _qend(*k) <= Ey]
                        add(tag, "USD", pd.Timestamp(yy, 1, 1), Ey, share * sum(qsale[k] for k in qs), filed, form, y, fp, accn)
                for tag, share in BS.items():               # balance sheet: period end + prior year end
                    if i == 2 and tag == "LongTermDebtCurrent":
                        continue                            # "no short-term debt -> no tag"
                    add(tag, "USD", None, E, share * bs_scale[(y, q)], filed, form, y, fp, accn)
                    if (y - 1, 4) in bs_scale:
                        add(tag, "USD", None, _qend(y - 1, 4), share * bs_scale[(y - 1, 4)], filed, form, y, fp, accn)
                cover = filed - pd.Timedelta(days=5)
                sh = shares0 * (2 if split_date is not None and cover >= split_date else 1)
                sh_bs = shares0 * (2 if split_date is not None and E >= split_date else 1)
                k = 1e-3 if (i == 5 and y == 2016) else 1.0              # both share counts reported in thousands
                add("CommonStockSharesOutstanding", "shares", None, E, sh_bs * k, filed, form, y, fp, accn)
                add("EntityCommonStockSharesOutstanding", "shares", None, cover,
                    100 if (i == 4 and y == 2015) else sh * k, filed, form, y, fp, accn)   # wrong cover-page count
                truth.append(dict(cik=cik, datadate=E, sale_q=qsale[(y, q)]))
        if i == 1:   # restatement: the 2015 annual figure is raised by a 10-K/A in 2016-09 and must be ignored
            facts["Revenues"]["USD"].append(dict(start="2015-01-01", end="2015-12-31", val=1e18,
                                                 filed="2016-09-30", form="10-K/A", fy=2015, fp="FY", accn="x"))
        docs.append({"cik": str(cik) if i % 2 else cik, "entityName": f"FIRM {i}",   # cik is sometimes a string in real data
                     "facts": {"us-gaap": {k: {"units": v} for k, v in facts.items() if k != "EntityCommonStockSharesOutstanding"},
                               "dei": {"EntityCommonStockSharesOutstanding": {"units": facts["EntityCommonStockSharesOutstanding"]}}}})
        # daily prices (yfinance convention: Close adjusted for later splits)
        days = pd.bdate_range("2007-01-01", "2021-03-31")
        me_true = size * np.exp(0.6 + 0.5 * quality + np.cumsum(rng.normal(0, 0.01, len(days)))) * 1e6
        shares_actual = np.where((split_date is not None) & (days >= (split_date or days[-1])), 2 * shares0, shares0)
        price_actual = me_true / shares_actual
        F = np.where((split_date is not None) & (days < (split_date or days[0])), 2.0, 1.0)
        spl = np.where(days == split_date, 2.0, 0.0) if split_date is not None else np.zeros(len(days))
        daily.append(pd.DataFrame({"ticker": f"T{i}", "date": days, "close": price_actual / F,
                                   "adj_close": price_actual / F, "split": spl, "me_true": me_true / 1e6}))
        tick.append((cik, f"FIRM {i}", f"T{i}", "NYSE" if i % 2 else "Nasdaq"))
    tickers = pd.DataFrame(tick, columns=["cik", "name", "ticker", "exchange"])
    sic = pd.DataFrame({"cik": tickers["cik"], "sic": [2834, 7372, 3571, 6020][0:1] * 0 + [[2834, 7372, 3571, 6020][k % 4] for k in range(n_firms)], "name": tickers["name"]})
    return docs, pd.DataFrame(truth), pd.concat(daily, ignore_index=True), tickers, sic
