"""Step 8: monthly cross-sectional GBMs; rotating five blocks gives every firm an out-of-sample prediction."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from .config import LGB_PARAMS, MAX_TREES, EARLY_STOP, N_BLOCKS, SEED
from .features import FEATURES


def assign_blocks(n: int, date: pd.Timestamp, seed: int = SEED) -> np.ndarray:
    """Randomly assign block numbers 0..4 to the n firms of a month. Reproducible for a given month and seed."""
    rng = np.random.default_rng([seed, date.year, date.month])
    return rng.permutation(np.arange(n) % N_BLOCKS)   # blocks of near-equal size


def fit_block(X, y, block, b):
    """Train one LightGBM model with block b as test, block b+1 as validation and the rest as training.
    Returns (model, best_iteration, train/validation/test boolean masks)."""
    te = block == b
    va = block == (b + 1) % N_BLOCKS
    tr = ~(te | va)
    dtr = lgb.Dataset(X.loc[tr], y.loc[tr], free_raw_data=False)
    dva = lgb.Dataset(X.loc[va], y.loc[va], reference=dtr)
    model = lgb.train(LGB_PARAMS, dtr, num_boost_round=MAX_TREES, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    it = model.best_iteration or model.current_iteration()
    return model, it, tr, va, te


def run_month(dm: pd.DataFrame, target: str, X: pd.DataFrame | None = None):
    """
    Inputs  dm : one month's cross-section (permno, gvkey, date, me, be, at, sale, debt, target)
            X  : the 97 features for that month (same index as dm)
    Outputs pred_df : one row per firm
                      permno, gvkey, date, block, y (actual log multiple), pred (out-of-sample),
                      ybar_train (training mean of y), mebar_train (training mean of me), n_trees
            shap_df : one row per feature - sum of normalized |SHAP| percentages over test rows, and the row count
    """
    date = dm["date"].iloc[0]
    y = dm[target]
    block = assign_blocks(len(dm), date)
    out = pd.DataFrame({"permno": dm["permno"].values, "gvkey": dm["gvkey"].values,
                        "date": date, "block": block, "y": y.values,
                        "pred": np.nan, "ybar_train": np.nan, "mebar_train": np.nan,
                        "n_trees": 0}, index=dm.index)
    shap_sum = np.zeros(len(FEATURES))
    for b in range(N_BLOCKS):
        model, it, tr, va, te = fit_block(X, y, block, b)
        out.loc[te, "pred"] = model.predict(X.loc[te], num_iteration=it)
        out.loc[te, "ybar_train"] = y.loc[tr].mean()
        out.loc[te, "mebar_train"] = dm.loc[tr, "me"].mean()
        out.loc[te, "n_trees"] = it
        # TreeSHAP: drop the last column (base value); scale each row's |SHAP| to sum to 100 (footnote 16)
        c = np.abs(model.predict(X.loc[te], num_iteration=it, pred_contrib=True)[:, :-1])
        tot = c.sum(axis=1, keepdims=True)
        shap_sum += np.nansum(np.divide(c, tot, where=tot > 0, out=np.zeros_like(c)) * 100, axis=0)
    shap_df = pd.DataFrame({"date": date, "feature": FEATURES,
                            "shap_sum": shap_sum, "n": len(dm)})
    return out.reset_index(drop=True), shap_df


def _one(args):
    dm, Xm, target = args
    return run_month(dm, target, Xm)


def run_all(panel: pd.DataFrame, X: pd.DataFrame, target: str, n_jobs: int = 1,
            min_firms: int = 100):
    """Loop over all months. Returns (preds, shap). n_jobs > 1 parallelizes across months.
    min_firms: skip months with too few firms (assumption, to keep leaves populated)."""
    if not isinstance(X["industry"].dtype, pd.CategoricalDtype):    # industry must be categorical
        X = X.copy()
        X["industry"] = pd.Categorical(X["industry"], categories=list(range(1, 50)))
    tasks = []
    for date, idx in panel.groupby("date").groups.items():
        if len(idx) >= min_firms:
            tasks.append((panel.loc[idx], X.loc[idx], target))
    if n_jobs == 1:
        res = []
        for i, t in enumerate(tasks):
            res.append(_one(t))
            if (i + 1) % 12 == 0:
                print(f"{target}: {i + 1}/{len(tasks)} months done")
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(n_jobs) as ex:
            res = list(ex.map(_one, tasks, chunksize=4))
    preds = pd.concat([r[0] for r in res], ignore_index=True)
    shap = pd.concat([r[1] for r in res], ignore_index=True)
    return preds, shap
