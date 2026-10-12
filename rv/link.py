"""Step 4: link CRSP (permno) and Compustat (gvkey) through the CCM table and, for each month,
attach the latest quarterly report available at that time."""
import pandas as pd


def merge_crsp_comp(crsp: pd.DataFrame, comp: pd.DataFrame, link: pd.DataFrame,
                    max_staleness_days: int = 365) -> pd.DataFrame:
    """
    Inputs  crsp : output of build_crsp_monthly (permno, date, me, ...)
            comp : output of build_comp_quarterly (gvkey, avail, accounting variables ...)
          link : gvkey, permno, linkdt, linkenddt
    Output  one row per permno-month: CRSP fields + the latest available report
    """
    lk = link.copy()
    lk["linkenddt"] = pd.to_datetime(lk["linkenddt"]).fillna(pd.Timestamp("2099-12-31"))
    df = crsp.merge(lk[["gvkey", "permno", "linkdt", "linkenddt"]], on="permno", how="inner")
    df = df[(df["date"] >= df["linkdt"]) & (df["date"] <= df["linkenddt"] + pd.offsets.MonthEnd(0))]
    df = df.drop(columns=["linkdt", "linkenddt"]).drop_duplicates(["permno", "date"])

    # merge_asof: for each (gvkey, date), the most recent report with avail <= date
    left = df.sort_values("date")
    right = comp.sort_values("avail")
    out = pd.merge_asof(left, right, left_on="date", right_on="avail", by="gvkey",
                        direction="backward",
                        tolerance=pd.Timedelta(days=max_staleness_days))
    out = out.dropna(subset=["datadate"])         # drop months with no available report
    return out.sort_values(["permno", "date"]).reset_index(drop=True)
