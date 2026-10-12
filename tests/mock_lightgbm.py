"""Test helper only: an sklearn-based stand-in for the lightgbm interface when lightgbm is not installed."""
import sys, types
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

mod = types.ModuleType("lightgbm")

def _prep(X):
    X = X.copy()
    for c in X.columns:
        if str(X[c].dtype) == "category":
            X[c] = X[c].cat.codes
    return X.astype(float).fillna(0).to_numpy()

class Dataset:
    def __init__(self, data, label=None, reference=None, free_raw_data=True, **kw):
        self.data, self.label = data, label

class Booster:
    def __init__(self, m): self.m, self.best_iteration = m, m.n_estimators
    def current_iteration(self): return self.m.n_estimators
    def predict(self, X, num_iteration=None, pred_leaf=False, pred_contrib=False):
        A = _prep(X)
        if pred_leaf:
            return self.m.apply(A).astype(int)
        if pred_contrib:
            rng = np.random.default_rng(0)
            c = rng.normal(0, 0.1, (len(A), A.shape[1] + 1)); c[:, -1] = self.m.init_.constant_[0][0]
            return c
        return self.m.predict(A)

def train(params, dtrain, num_boost_round=100, valid_sets=None, callbacks=None):
    m = GradientBoostingRegressor(n_estimators=30, learning_rate=params["learning_rate"],
                                  max_leaf_nodes=params["num_leaves"],
                                  min_samples_leaf=params["min_data_in_leaf"], random_state=0)
    m.fit(_prep(dtrain.data), np.asarray(dtrain.label))
    return Booster(m)

def early_stopping(n, verbose=True): return lambda env: None

mod.Dataset, mod.train, mod.early_stopping, mod.Booster = Dataset, train, early_stopping, Booster
sys.modules["lightgbm"] = mod
