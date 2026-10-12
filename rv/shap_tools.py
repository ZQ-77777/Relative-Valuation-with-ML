"""Step 11: aggregate monthly SHAP values into Table 7 (variables), Table 8 (categories) and Table 9 (by decade)."""
import pandas as pd
from .features import CATEGORY


def variable_importance(shap: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Input: shap output of run_all. Output: variables ranked by mean |SHAP|% with cumulative CSHAP."""
    tot = shap.groupby("feature")[["shap_sum", "n"]].sum()
    imp = (tot["shap_sum"] / tot["n"]).sort_values(ascending=False).rename("SHAP")
    t = imp.to_frame()
    t["CSHAP"] = t["SHAP"].cumsum()
    t["Category"] = t.index.map(CATEGORY)
    return t.head(top).round(1)


def category_importance(shap: pd.DataFrame) -> pd.DataFrame:
    """Table 8: category importance, full sample and by decade (sum of SHAP within each category)."""
    s = shap.copy()
    s["Category"] = s["feature"].map(CATEGORY)
    s["decade"] = (s["date"].dt.year // 10 * 10).astype(str)
    def agg(x):
        n = x.drop_duplicates("date")["n"].sum()   # total test observations across months
        return x.groupby("Category")["shap_sum"].sum() / n
    full = agg(s).rename("SHAP")
    by_dec = s.groupby("decade").apply(agg).T
    out = pd.concat([full, by_dec], axis=1).sort_values("SHAP", ascending=False)
    if {"1980", "2010"} <= set(out.columns):
        out["Chg"] = out["2010"] - out["1980"]
    return out.round(1)
