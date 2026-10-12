"""Step 6: targets (three log multiples) and sample filters (Table 1)."""
import numpy as np
import pandas as pd
from .config import SIZE_PCTL, START, END


def add_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Input   panel with me (market equity), be, at, sale, dltt, dlc, pstk ($ millions)
    Output  adds debt, V, m2b, v2a, v2s, lnm2b, lnv2a, lnv2s;
            keeps rows where all three multiples are positive and finite (Table 1, row 2)
    Enterprise value V = market equity + book debt (dltt + dlc + pstk), i.e. debt_book_value in Table A.1.
    This definition is an assumption; the paper's exact definition is in Online Appendix IA.4.
    """
    d = panel.copy()
    d["debt"] = d[["dltt", "dlc", "pstk"]].fillna(0).sum(axis=1)
    d["V"] = d["me"] + d["debt"]
    d["m2b"] = d["me"] / d["be"]
    d["v2a"] = d["V"] / d["at"]
    d["v2s"] = d["V"] / d["sale"]
    mult = d[["m2b", "v2a", "v2s"]]
    ok = (mult > 0).all(axis=1) & np.isfinite(mult).all(axis=1)
    d = d[ok].copy()
    for m in ["m2b", "v2a", "v2s"]:
        d["ln" + m] = np.log(d[m])
    return d


def size_filter(d: pd.DataFrame, pctl: float = SIZE_PCTL, start: str = START,
                end: str = END) -> pd.DataFrame:
    """Each month, drop rows with at, be or sale below the monthly pctl quantile (Table 1, row 3); keep [start, end]."""
    d = d[(d["date"] >= start) & (d["date"] <= end)]
    thr = d.groupby("date")[["at", "be", "sale"]].transform(lambda s: s.quantile(pctl))
    keep = (d["at"] >= thr["at"]) & (d["be"] >= thr["be"]) & (d["sale"] >= thr["sale"])
    return d[keep].reset_index(drop=True)


def table1(raw_n: tuple, after_targets: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """Sample-selection table comparable to the paper's Table 1. raw_n = (firm-months, firms)."""
    rows = [("NYSE/NASDAQ/AMEX common stocks", *raw_n),
            ("Target multiples positive and non-missing", len(after_targets), after_targets["gvkey"].nunique()),
            ("Size variables above monthly 10th percentile", len(final), final["gvkey"].nunique())]
    return pd.DataFrame(rows, columns=["Step", "Firm-months", "Firms"])
