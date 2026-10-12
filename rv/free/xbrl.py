"""Free data, step 2: convert SEC XBRL companyfacts into a table with the same layout as Compustat quarterly.

Approach
  1. each Compustat item maps to candidate XBRL tags (STOCK_TAGS, FLOW_TAGS); the first available tag wins;
  2. the same number reappears as a comparative in later filings and may be restated.
     For each (tag, start, end) only the first-filed value is kept - what investors actually saw -
     so the data are point-in-time, with no look-ahead from restatements;
  3. flow items are converted to trailing twelve months (TTM):
        annual period:    TTM = annual value
        interim period:   TTM = current YTD + prior fiscal year - prior-year YTD of the same length
     which handles both income statements (3-month and YTD values) and cash-flow statements (YTD only);
  4. each period becomes available at its 10-Q/10-K filing date, more precise than the paper's uniform 3-month lag.
"""
import numpy as np
import pandas as pd

from ..compustat import finalize_quarterly
from ..config import INFO_LAG_MONTHS

FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT", "10-QT"}

# stock items (balance sheet, a point in time): Compustat item -> candidate us-gaap tags (in priority order)
STOCK_TAGS = {
    "at":     ["Assets"],
    "lt":     ["Liabilities"],
    "act":    ["AssetsCurrent"],
    "lct":    ["LiabilitiesCurrent"],
    "che":    ["CashAndCashEquivalentsAtCarryingValue",
               "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"],
    "seq":    ["StockholdersEquity"],
    "pstk":   ["PreferredStockValue"],
    "txditc": ["DeferredTaxLiabilitiesNoncurrent", "DeferredIncomeTaxLiabilitiesNet"],
    "ppent":  ["PropertyPlantAndEquipmentNet"],
    "invt":   ["InventoryNet"],
    "rect":   ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "ap":     ["AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"],
    "dltt":   ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations",
               "LongTermNotesPayable", "SeniorLongTermNotes"],
    "dlc":    ["DebtCurrent", "LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent",
               "ShortTermBorrowings"],
    "mib":    ["MinorityInterest"],
}
# flow items (income and cash-flow statements, a period)
FLOW_TAGS = {
    "sale":   ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
               "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax",
               "RegulatedAndUnregulatedOperatingRevenue", "RealEstateRevenueNet",
               "RevenuesNetOfInterestExpense", "InterestAndDividendIncomeOperating"],
    "cogs":   ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
               "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization", "CostOfServices"],
    "gp":     ["GrossProfit"],                                   # helper: cogs = sale - gp when cogs is missing
    "xsga":   ["SellingGeneralAndAdministrativeExpense"],
    "ga":     ["GeneralAndAdministrativeExpense"],               # helper: SG&A reported in parts is summed
    "sm":     ["SellingAndMarketingExpense", "SellingExpense", "MarketingAndAdvertisingExpense"],
    "xrd":    ["ResearchAndDevelopmentExpense"],
    "dp":     ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
               "DepreciationAmortizationAndAccretionNet", "Depreciation"],
    "xint":   ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt",
               "InterestAndDebtExpense"],
    "oiadp":  ["OperatingIncomeLoss"],
    "pi":     ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
               "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "txt":    ["IncomeTaxExpenseBenefit"],
    "ib":     ["IncomeLossFromContinuingOperations", "NetIncomeLoss"],
    "ibcom":  ["NetIncomeLossAvailableToCommonStockholdersBasic", "NetIncomeLoss"],
    "nopi":   ["NonoperatingIncomeExpense"],
    "mii":    ["NetIncomeLossAttributableToNoncontrollingInterest"],
    "capx":   ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "oancf":  ["NetCashProvidedByUsedInOperatingActivities",
               "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "dv":     ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
}
SHARE_TAGS = [("dei", "EntityCommonStockSharesOutstanding"),
              ("us-gaap", "CommonStockSharesOutstanding")]


MIN_END = pd.Timestamp("2006-01-01")   # earlier data are not needed (sample starts 2012; lags need a few years)


def facts_long(doc: dict) -> pd.DataFrame:
    """
    Input   one firm's companyfacts JSON (dict)
    Output  long table: cik, tag, unit, start, end, val, filed, form, fy, fp (only tags used in the mappings)
    """
    wanted = {t for v in {**STOCK_TAGS, **FLOW_TAGS}.values() for t in v}
    rows = []
    cik = int(doc.get("cik") or 0)   # cik may be a number, a string or missing
    cols = ["cik", "tag", "unit", "start", "end", "val", "filed", "form", "fy", "fp", "accn"]
    if not cik or not doc.get("facts"):
        return pd.DataFrame(columns=cols)
    facts = doc.get("facts", {})
    for tax, tag in [("us-gaap", t) for t in wanted] + SHARE_TAGS:
        node = facts.get(tax, {}).get(tag)
        if not node:
            continue
        for unit, lst in node.get("units", {}).items():
            if unit not in ("USD", "shares"):
                continue
            for f in lst:
                if f.get("form") in FORMS:
                    rows.append((cik, tag, unit, f.get("start"), f["end"], f["val"],
                                 f["filed"], f["form"], f.get("fy"), f.get("fp"), f.get("accn")))
    df = pd.DataFrame(rows, columns=cols)
    for c in ["start", "end", "filed"]:
        # a few filers enter invalid dates (e.g. 0202-02-01): coerce to NaT, then drop the row
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["end", "filed"])
    df = df[(df["end"] >= MIN_END) & (df["end"] <= df["filed"])
            & (df["start"].isna() | (df["start"] <= df["end"]))]
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce")
    return df


def _first_filed(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Keep only the first-filed value for each item and period (point-in-time)."""
    return df.sort_values("filed").drop_duplicates(keys, keep="first")


def periods(f: pd.DataFrame) -> pd.DataFrame:
    """Reporting periods: one row per (cik, period end) with first filing date, fiscal year fyearq and quarter fqtr.
    Based on Assets: a period's total assets first appear in that period's own 10-Q/10-K."""
    a = f[(f["tag"] == "Assets") & f["start"].isna() & f["form"].isin({"10-Q", "10-K", "10-KT"})]
    a = _first_filed(a, ["cik", "end"])
    p = a.rename(columns={"end": "datadate"})[["cik", "datadate", "filed", "fy", "fp"]]
    p["fqtr"] = p["fp"].map({"Q1": 1, "Q2": 2, "Q3": 3, "FY": 4})
    p = p.dropna(subset=["fqtr"]).rename(columns={"fy": "fyearq"})
    return p.drop(columns="fp")


def stock_values(f: pd.DataFrame) -> pd.DataFrame:
    """Stock items: period-end values, combining candidate tags by priority. Output cik, datadate, at, lt, ..."""
    s = f[f["start"].isna() & (f["unit"] == "USD")]
    s = _first_filed(s, ["cik", "tag", "end"])
    wide = s.pivot_table(index=["cik", "end"], columns="tag", values="val", aggfunc="first",
                         observed=True)
    out = pd.DataFrame(index=wide.index)
    for field, tags in STOCK_TAGS.items():
        col = pd.Series(np.nan, index=wide.index)
        for t in tags:
            if t in wide:
                col = col.fillna(wide[t])
        out[field] = col
    return out.reset_index().rename(columns={"end": "datadate"})


def ttm_values(f: pd.DataFrame) -> pd.DataFrame:
    """
    Flow items converted to TTM. Output cik, datadate, sale, cogs, ...
    m = period length in months (3, 6, 9, 12). The day before the start of YTD value y_E is the prior fiscal year end P.
    TTM(E) = A12(E)                         if a 12-month value ending at E exists
           = y_E + A12(P) - y_{E-1yr}       otherwise (prior-year YTD of the same length)
    """
    d = f[f["start"].notna() & (f["unit"] == "USD")].copy()
    d["m"] = ((d["end"] - d["start"]).dt.days / 30.44).round().astype(int)
    d = d[d["m"].isin([3, 6, 9, 12])]
    d = _first_filed(d, ["cik", "tag", "start", "end"])
    a12 = d[d["m"] == 12][["cik", "tag", "end", "val"]].drop_duplicates(["cik", "tag", "end"])
    # for each (cik, tag, end) keep the longest YTD value (10-Qs report both 3-month and YTD; use YTD)
    y = (d[d["m"] < 12].sort_values("m").drop_duplicates(["cik", "tag", "end"], keep="last")
         [["cik", "tag", "start", "end", "m", "val"]])
    y["P"] = y["start"] - pd.Timedelta(days=1)
    y = y.merge(a12.rename(columns={"end": "P", "val": "a_prev"}), on=["cik", "tag", "P"], how="left")
    # prior-year YTD of the same length, allowing +/-10 days (52/53-week fiscal years)
    prior = y[["cik", "tag", "m", "end", "val"]].rename(columns={"end": "end_prev", "val": "y_prev"})
    y["target"] = y["end"] - pd.Timedelta(days=365)
    y = pd.merge_asof(y.sort_values("target"), prior.sort_values("end_prev"),
                      left_on="target", right_on="end_prev", by=["cik", "tag", "m"],
                      direction="nearest", tolerance=pd.Timedelta(days=10))
    y["ttm"] = y["val"] + y["a_prev"] - y["y_prev"]
    ttm = pd.concat([a12.rename(columns={"val": "ttm"}).assign(src=0),
                     y[["cik", "tag", "end", "ttm"]].assign(src=1)])
    ttm = ttm.dropna(subset=["ttm"]).sort_values("src").drop_duplicates(["cik", "tag", "end"])
    wide = ttm.pivot_table(index=["cik", "end"], columns="tag", values="ttm", aggfunc="first",
                           observed=True)
    out = pd.DataFrame(index=wide.index)
    for field, tags in FLOW_TAGS.items():
        col = pd.Series(np.nan, index=wide.index)
        for t in tags:
            if t in wide:
                col = col.fillna(wide[t])
        out[field] = col
    return out.reset_index().rename(columns={"end": "datadate"})


def shares_table(f: pd.DataFrame) -> pd.DataFrame:
    """
    Shares outstanding: cik, sh_end (date of the count), sh_filed, shares. One row per filing.
    Two sources are cross-checked:
      dei:EntityCommonStockSharesOutstanding  cover-page count (most recent, but occasionally wrong, e.g. 100 shares before a spin-off)
      us-gaap:CommonStockSharesOutstanding    balance-sheet-date count
    If they differ by more than 5x the balance-sheet count is used; counts below 10,000 are treated as errors.
    Share classes reported separately (dual-class firms) are summed for the same date.
    """
    s = f[(f["unit"] == "shares") & f["tag"].isin([t for _, t in SHARE_TAGS])].copy()
    s["tag"] = s["tag"].astype(str)
    s = (s.groupby(["cik", "accn", "filed", "tag", "end"], as_index=False, observed=True)["val"].sum())
    s = s.sort_values("end").groupby(["cik", "accn", "tag"], observed=True).tail(1)   # latest date per filing
    w = s.pivot_table(index=["cik", "accn", "filed"], columns="tag", values=["val", "end"],
                      aggfunc="first", observed=True)
    dei_v = w[("val", "EntityCommonStockSharesOutstanding")] if ("val", "EntityCommonStockSharesOutstanding") in w else pd.Series(np.nan, index=w.index)
    gap_v = w[("val", "CommonStockSharesOutstanding")] if ("val", "CommonStockSharesOutstanding") in w else pd.Series(np.nan, index=w.index)
    dei_e = w[("end", "EntityCommonStockSharesOutstanding")] if ("end", "EntityCommonStockSharesOutstanding") in w else pd.Series(pd.NaT, index=w.index)
    gap_e = w[("end", "CommonStockSharesOutstanding")] if ("end", "CommonStockSharesOutstanding") in w else pd.Series(pd.NaT, index=w.index)
    dei_v, gap_v = dei_v.where(dei_v >= 1e4), gap_v.where(gap_v >= 1e4)
    ratio = dei_v / gap_v
    use_dei = dei_v.notna() & (gap_v.isna() | ratio.between(0.2, 5))
    out = pd.DataFrame({"shares": np.where(use_dei, dei_v, gap_v),
                        "sh_end": np.where(use_dei, dei_e, gap_e)}, index=w.index).reset_index()
    out["sh_end"] = pd.to_datetime(out["sh_end"])
    out = out.dropna(subset=["shares", "sh_end"]).rename(columns={"filed": "sh_filed"})
    return out[["cik", "sh_end", "sh_filed", "shares"]].reset_index(drop=True)


def build_quarterly(facts: pd.DataFrame, avail_mode: str = "filed") -> pd.DataFrame:
    """
    Input   output of facts_long (any number of firms)
    Output  the same columns as compustat.build_comp_quarterly; gvkey is the CIK as a string
    """
    p = periods(facts)
    q = (p.merge(stock_values(facts), on=["cik", "datadate"], how="left")
          .merge(ttm_values(facts), on=["cik", "datadate"], how="left"))
    # ---- units: XBRL amounts are in dollars; Compustat and market equity are in $ millions ----
    money = [c for c in list(STOCK_TAGS) + list(FLOW_TAGS) if c in q]
    q[money] = q[money] / 1e6
    # ---- drop periods with clear unit errors: total assets 100x away from the firm's median ----
    g = q.groupby("cik")
    r_at = q["at"] / g["at"].transform("median")
    r_eq = q["seq"].abs() / g["seq"].transform(lambda x: x.abs().median())   # equity may be near 0: check upward only
    q = q[~((r_at > 100) | (r_at < 0.01) | (r_eq > 100))].copy()
    # ---- fill-ins and definitional alignment ----
    # (a) XBRL omits zero items (no debt -> no debt tag) where Compustat records 0.
    #     When total assets are reported, set these "absent means zero" items to 0
    has = q["at"].notna()
    for c in ["dltt", "dlc", "pstk", "mib", "invt", "xrd", "dv"]:
        q.loc[has, c] = q.loc[has, c].fillna(0)
    q.loc[has & (q["dltt"] + q["dlc"] == 0), "xint"] = q["xint"].fillna(0)   # no debt -> no interest
    # (b) missing cost of goods sold = sales - gross profit
    q["cogs"] = q["cogs"].fillna(q["sale"] - q["gp"])
    # (c) SG&A reported in parts is summed; Compustat XSGA includes R&D, so add xrd
    q["xsga"] = q["xsga"].fillna(q["ga"] + q["sm"].fillna(0)) + q["xrd"].fillna(0)
    # (d) missing operating income: Compustat definition sale - cogs - xsga - dp first,
    #     then pretax income + interest - nonoperating income as an approximation
    q["oiadp"] = (q["oiadp"].fillna(q["sale"] - q["cogs"] - q["xsga"] - q["dp"])
                            .fillna(q["pi"] + q["xint"].fillna(0) - q["nopi"].fillna(0)))
    # items with no direct XBRL counterpart
    q["ceq"] = q["seq"] - q["pstk"].fillna(0)
    q["lt"] = q["lt"].fillna(q["at"] - q["seq"] - q["mib"].fillna(0))
    q["oibdp"] = q["oiadp"] + q["dp"]
    q["ibadj"] = q["ib"]
    q["icapt"] = q["dltt"].fillna(0) + q["ceq"] + q["pstk"].fillna(0) + q["mib"].fillna(0)
    q["gvkey"] = q["cik"].astype(int).astype(str).str.zfill(10)
    if avail_mode == "filed":
        q["avail"] = q["filed"] + pd.offsets.MonthEnd(0)      # usable from the month-end of the filing month
    else:
        q["avail"] = q["datadate"] + pd.offsets.MonthEnd(INFO_LAG_MONTHS)
    return finalize_quarterly(q)


BASE_VARS = ["at", "lt", "act", "lct", "che", "seq", "ppent", "invt", "rect", "ap", "dltt", "dlc",
             "sale", "cogs", "xsga", "xrd", "dp", "xint", "oiadp", "pi", "txt", "ib", "oancf", "capx"]


def coverage(q: pd.DataFrame, by: pd.Series | None = None):
    """Coverage (share non-missing) of base variables, to see which tag mappings need work.
    If by is given, computed by group (e.g. financial vs. non-financial)."""
    cols = [c for c in BASE_VARS if c in q]
    if by is None:
        return q[cols].notna().mean().sort_values().round(3)
    return q[cols].notna().groupby(by).mean().T.round(3).sort_values(by=list(by.dropna().unique())[0])
