"""Step 10: long-short portfolios sorted on valuation errors and Fama-French six-factor regressions (Table 3)."""
import numpy as np
import pandas as pd


def long_short(d: pd.DataFrame, signal: str = "eps") -> pd.Series:
    """
    Input   d : one row per permno-month with date, exchcd, me, ret_next and the signal eps
    Output  long-short returns indexed by formation month (quintile 5 - quintile 1, value-weighted)
            Q1 = most negative eps = most overvalued; Q5 = most positive eps = most undervalued
    Breakpoints use NYSE stocks only (exchcd == 1; paper footnote 12).
    """
    def one_month(m):
        m = m.dropna(subset=[signal, "ret_next", "me"])
        nyse = m.loc[m["exchcd"] == 1, signal]
        if len(nyse) < 20:
            return np.nan
        bps = nyse.quantile([0.2, 0.4, 0.6, 0.8]).values
        q = np.searchsorted(bps, m[signal].values, side="right") + 1
        r = {}
        for k in (1, 5):
            g = m[q == k]
            r[k] = np.average(g["ret_next"], weights=g["me"]) if len(g) else np.nan
        return r[5] - r[1]
    return d.groupby("date")[[signal, "ret_next", "me", "exchcd"]].apply(one_month).rename("ls")


def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int = 3):
    """OLS with Newey-West standard errors. X must include a constant. Returns (coef, se, t, residuals)."""
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xe[L:].T @ Xe[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return b, se, b / se, e


def factor_test(ls: pd.Series, ff: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """
    Inputs  ls : output of long_short (indexed by formation month)
            ff : date, mktrf, smb, hml, rmw, cma, umd (decimals)
    Output  table: mean, annualized Sharpe ratio, FF6 alpha, factor loadings, t-stats, adjusted R2, information ratio
    Note: portfolios formed in month t earn returns in t+1, so they are aligned with month t+1 factors.
    """
    r = ls.dropna().rename("ls").to_frame()
    r.index = r.index + pd.offsets.MonthEnd(1)          # align to the return month
    m = r.join(ff.set_index("date"), how="inner")
    y = m["ls"].to_numpy() * 100                         # percent, as in Table 3
    facs = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
    rows = {}
    # mean: regression on a constant, also with Newey-West
    b0, se0, t0, _ = newey_west_ols(y, np.ones((len(y), 1)), lags)
    rows["Mean Const"] = (b0[0], t0[0])
    rows["Sharpe (annualized)"] = (y.mean() / y.std(ddof=1) * np.sqrt(12), np.nan)
    X = np.column_stack([np.ones(len(y)), m[facs].to_numpy() * 100])
    b, se, t, e = newey_west_ols(y, X, lags)
    for i, name in enumerate(["FF6 Const"] + facs):
        rows[name] = (b[i], t[i])
    n, k = X.shape
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    rows["Adj R2"] = (1 - (1 - r2) * (n - 1) / (n - k), np.nan)
    rows["Information ratio (annualized)"] = (b[0] / e.std(ddof=k) * np.sqrt(12), np.nan)
    return pd.DataFrame(rows, index=["estimate", "t-stat"]).T.round(3)
