"""Step 12: GBM peer weights (Appendix A.3) and single-firm SHAP decompositions (Table 5, Figures 4 and 5)."""
import numpy as np
import pandas as pd
from .config import LGB_PARAMS
from .features import FEATURES
from .train import assign_blocks, fit_block


def peer_weights(model, X_train, X_test, lr=LGB_PARAMS["learning_rate"], num_iteration=None):
    """
    Inputs  model   : trained LightGBM Booster
            X_train : training features of that model (N rows)
            X_test  : features of the firms to value (S rows)
    Output  K : S x N array. Row s holds test firm s's weights on the N training firms; each row sums to 1.
                K @ y_train == model.predict(X_test)  (paper eq. A.11: k = v'K)
    Code variables vs. paper notation:
      leaf ids L_tr[:, i] -> D_i; W -> W_i; P -> P_i; G -> G_i; same_te -> L_i; K -> transpose of K
    """
    L_tr = model.predict(X_train, pred_leaf=True, num_iteration=num_iteration)
    L_te = model.predict(X_test, pred_leaf=True, num_iteration=num_iteration)
    L_tr, L_te = np.atleast_2d(L_tr), np.atleast_2d(L_te)
    N, S = L_tr.shape[0], L_te.shape[0]
    I = np.eye(N)
    G = np.full((N, N), 1.0 / N)        # G_0 = P_0 = (1/N)·1
    K = np.full((S, N), 1.0 / N)
    for i in range(L_tr.shape[1]):
        same = (L_tr[:, i][:, None] == L_tr[:, i][None, :]).astype(float)       # D_i
        W = same / same.sum(axis=1, keepdims=True)                              # eq. A.5
        P = lr * W @ (I - G)                                                    # eq. A.6
        same_te = (L_te[:, i][:, None] == L_tr[:, i][None, :]).astype(float)    # L_i
        K += (same_te / same_te.sum(axis=1, keepdims=True)) @ P                 # eq. A.10
        G += P                                                                  # eq. A.8
    return K


def firm_report(dm: pd.DataFrame, X: pd.DataFrame, target: str, permno: int,
                names: pd.Series | None = None, top: int = 10):
    """
    For one firm in month dm, retrain the model of its test block and return:
      peers   : all training firms with permno, name, target, weight, cumulative weight (descending) - cf. Table 5
      shap    : the firm's raw (signed) SHAP contributions - cf. Figure 5
      summary : actual, predicted, peer-weighted average and training mean
    names : optional company names indexed by permno
    """
    date = dm["date"].iloc[0]
    y = dm[target]
    block = assign_blocks(len(dm), date)
    pos = np.flatnonzero(dm["permno"].to_numpy() == permno)
    if len(pos) == 0:
        raise KeyError(f"{permno} is not in this month's sample")
    b = block[pos[0]]
    model, it, tr, va, te = fit_block(X, y, block, b)
    x0 = X.iloc[[pos[0]]]
    K = peer_weights(model, X.loc[tr], x0, num_iteration=it)[0]
    peers = pd.DataFrame({"permno": dm.loc[tr, "permno"].to_numpy(),
                          target: y.loc[tr].to_numpy(), "weight": K})
    if names is not None:
        peers.insert(1, "name", peers["permno"].map(names))
    peers = peers.sort_values("weight", ascending=False).reset_index(drop=True)
    peers["cumweight"] = peers["weight"].cumsum()
    contrib = model.predict(x0, num_iteration=it, pred_contrib=True)[0]
    shap = pd.Series(contrib[:-1], index=FEATURES).sort_values(key=np.abs, ascending=False)
    pred = float(model.predict(x0, num_iteration=it)[0])
    summary = {"actual": float(y.iloc[pos[0]]), "pred": pred,
               "weighted_avg": float(K @ y.loc[tr].to_numpy()),
               "base_value (training mean)": float(contrib[-1]), "n_trees": it}
    return peers, shap, summary


def value_new_firm(dm: pd.DataFrame, X: pd.DataFrame, target: str,
                   x_new: pd.DataFrame, denom: float, debt: float = 0.0, seed: int = 0):
    """
    Value a firm outside the sample (e.g. an IPO candidate or a private firm).
    Inputs  dm, X  : the latest month's full sample and features (the "market")
            x_new  : the new firm's 97 features (one row, same columns as X; use build_features)
            denom  : denominator of the multiple (be for lnm2b, total assets for lnv2a, sales for lnv2s)
            debt   : debt to subtract for enterprise-value multiples (0 for lnm2b)
    Output  dict: predicted multiple, implied equity value, peer table, SHAP contributions
    Method: train on 80% of firms with 20% for early stopping, then compute peer weights and SHAP.
    """
    rng = np.random.default_rng(seed)
    va = rng.random(len(dm)) < 0.2
    tr = ~va
    import lightgbm as lgb
    from .config import MAX_TREES, EARLY_STOP
    y = dm[target]
    dtr = lgb.Dataset(X.loc[tr], y.loc[tr], free_raw_data=False)
    dva = lgb.Dataset(X.loc[va], y.loc[va], reference=dtr)
    model = lgb.train(LGB_PARAMS, dtr, num_boost_round=MAX_TREES, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    it = model.best_iteration or model.current_iteration()
    pred = float(model.predict(x_new, num_iteration=it)[0])
    K = peer_weights(model, X.loc[tr], x_new, num_iteration=it)[0]
    peers = (pd.DataFrame({"permno": dm.loc[tr, "permno"].to_numpy(),
                           "name": dm.loc[tr, "comnam"].to_numpy() if "comnam" in dm else None,
                           target: y.loc[tr].to_numpy(), "weight": K})
               .sort_values("weight", ascending=False).reset_index(drop=True))
    contrib = model.predict(x_new, num_iteration=it, pred_contrib=True)[0]
    shap = pd.Series(contrib[:-1], index=FEATURES).sort_values(key=np.abs, ascending=False)
    multiple = float(np.exp(pred))
    return {"pred_log_multiple": pred, "multiple": multiple,
            "equity_value": multiple * denom - debt, "peers": peers, "shap": shap,
            "base_value": float(contrib[-1])}
