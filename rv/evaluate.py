"""Step 9: convert predicted log multiples to equity values and compute the Table 2 metrics."""
import numpy as np
import pandas as pd


def implied_equity(preds: pd.DataFrame, panel: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Inputs  preds : output of run_all (permno, date, pred ...)
            panel : panel with me, be, at, sale, debt
    Output  preds plus me (actual market value), M_hat (predicted), eps (valuation error = M_hat/me - 1)
          lnm2b: M_hat = exp(pred) × be
          lnv2a: M_hat = exp(pred) x at   - debt   (enterprise value first, then subtract debt)
          lnv2s: M_hat = exp(pred) × sale − debt
    """
    cols = ["permno", "date", "me", "be", "at", "sale", "debt"]
    d = preds.merge(panel[cols], on=["permno", "date"], how="left")
    mult = np.exp(d["pred"])
    if target == "lnm2b":
        d["M_hat"] = mult * d["be"]
    elif target == "lnv2a":
        d["M_hat"] = mult * d["at"] - d["debt"]
    elif target == "lnv2s":
        d["M_hat"] = mult * d["sale"] - d["debt"]
    else:
        raise ValueError(target)
    d["eps"] = d["M_hat"] / d["me"] - 1
    return d


def metrics(d: pd.DataFrame) -> pd.Series:
    """One row of Table 2. d needs y, pred, ybar_train, me, M_hat, mebar_train, eps."""
    d = d.dropna(subset=["pred", "M_hat", "me"])
    e = d["eps"]
    r2 = lambda a, ahat, abar: 1 - ((a - ahat) ** 2).sum() / ((a - abar) ** 2).sum()
    return pd.Series({
        "N": len(d),
        "med(|e|)": e.abs().median(), "avg(|e|)": e.abs().mean(),
        "med(e)": e.median(), "avg(e)": e.mean(), "sd(e)": e.std(),
        "rmse(y)": np.sqrt(((d["y"] - d["pred"]) ** 2).mean()),
        "rmse(M)": np.sqrt(((d["me"] - d["M_hat"]) ** 2).mean()),
        "r2oos(y)": r2(d["y"], d["pred"], d["ybar_train"]),
        "r2oos(M)": r2(d["me"], d["M_hat"], d["mebar_train"]),
        "rho(M,Mhat)": d["me"].corr(d["M_hat"]),
    })


def table2(results: dict) -> pd.DataFrame:
    """results = {"ML_lnm2b": d1, "IND_m2b": d2, ...}, each from implied_equity."""
    return pd.DataFrame({k: metrics(v) for k, v in results.items()}).T.round(2)
