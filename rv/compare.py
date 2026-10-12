"""Comparing the WRDS and free-data versions ("triangulation").

Paper 1980-2019 (Compustat)  <->  your WRDS version  <->  your free-data version 2012-2024 (XBRL)
Restricting both versions to a common period (e.g. 2012-2019) and common firms separates differences
due to the data source from differences due to the sample period.
"""
import numpy as np
import pandas as pd

from .evaluate import metrics


def restrict(d: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Keep observations dated within [start, end]."""
    return d[(d["date"] >= start) & (d["date"] <= end)]


def match_sources(wrds: pd.DataFrame, free: pd.DataFrame, company: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs  wrds    : implied_equity output of the WRDS version (gvkey, date, y, pred, eps ...)
            free    : implied_equity output of the free version (permno = CIK)
          company : comp.company (gvkey, cik)
    Output  the same firm-month side by side in both versions: *_w = WRDS, *_f = free data
    """
    c = company.dropna(subset=["cik"]).copy()
    c["cik"] = pd.to_numeric(c["cik"], errors="coerce").astype("Int64")
    w = wrds.merge(c[["gvkey", "cik"]], on="gvkey", how="inner")
    f = free.rename(columns={"permno": "cik"})
    f["cik"] = f["cik"].astype("Int64")
    keep = ["cik", "date", "y", "pred", "me", "M_hat", "eps"]
    return w[keep].merge(f[keep], on=["cik", "date"], suffixes=("_w", "_f"))


def compare_report(m: pd.DataFrame) -> pd.Series:
    """Agreement between the two versions on their common sample."""
    return pd.Series({
        "common firm-months": len(m),
        "common firms": m["cik"].nunique(),
        "target correlation (data agreement)": m["y_w"].corr(m["y_f"]),
        "median abs. target difference": (m["y_w"] - m["y_f"]).abs().median(),
        "prediction correlation": m["pred_w"].corr(m["pred_f"]),
        "valuation-error rank correlation": m["eps_w"].rank().corr(m["eps_f"].rank()),
        "med(|e|) WRDS": m["eps_w"].abs().median(),
        "med(|e|) free": m["eps_f"].abs().median(),
    }).round(3)


def side_by_side(results: dict) -> pd.DataFrame:
    """results = {"WRDS 1980-2019": d1, "WRDS 2012-2019": d2, "free 2012-2019": d3, ...} -> Table 2 metrics side by side"""
    return pd.DataFrame({k: metrics(v) for k, v in results.items()}).T.round(2)


def category_agreement(shap_a: pd.DataFrame, shap_b: pd.DataFrame) -> float:
    """Do the two versions rank variables alike? Spearman correlation of mean SHAP by variable."""
    imp = lambda s: s.groupby("feature")["shap_sum"].sum() / s.drop_duplicates("date")["n"].sum()
    a, b = imp(shap_a), imp(shap_b)
    j = a.index.intersection(b.index)
    return float(a[j].rank().corr(b[j].rank()))
