"""Method check (no financial data needed): python -m tests.validate_peer_weights

Experiment 1: with default settings, peer-weighted averages equal LightGBM predictions (the identity of Appendix A.3).
Experiment 2: row subsampling (bagging) or L2 regularization (lambda_l2) breaks the identity,
        because leaf values are no longer simple averages of in-leaf residuals - hence the default settings.
Requires the real lightgbm and scikit-learn.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import make_friedman1
from rv.peers import peer_weights

X, y = make_friedman1(n_samples=600, noise=1.0, random_state=0)
X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
Xtr, ytr, Xte = X.iloc[:500], y[:500], X.iloc[500:]

base = dict(objective="regression", learning_rate=0.1, num_leaves=31, min_data_in_leaf=20,
            verbose=-1, seed=0, deterministic=True)
settings = {
    "default (paper setting)": {},
    "bagging_fraction=0.5": dict(bagging_fraction=0.5, bagging_freq=1),
    "lambda_l2=10": dict(lambda_l2=10.0),
}
rows = []
for name, extra in settings.items():
    m = lgb.train({**base, **extra}, lgb.Dataset(Xtr, ytr), num_boost_round=100)
    K = peer_weights(m, Xtr, Xte, lr=base["learning_rate"])
    gap = np.abs(K @ ytr - m.predict(Xte)).max()
    rows.append((name, gap, np.abs(K.sum(axis=1) - 1).max(), (K < 0).mean()))
print(pd.DataFrame(rows, columns=["setting", "max|weighted avg - pred|", "max|sum of weights - 1|", "share of negative weights"]))
print("\nExpected: first row within ~1e-6 (LightGBM stores gradients in single precision); "
      "the other two rows clearly above 0 (around 0.1 or more).")
