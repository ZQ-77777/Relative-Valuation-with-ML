"""List the firm-months with the most extreme valuation errors to spot data errors (e.g. share-count or unit errors).
Usage: python check_outliers.py lnm2b"""
import sys
import pandas as pd
from rv.config import INTERIM, OUT
from rv.evaluate import implied_equity

t = sys.argv[1] if len(sys.argv) > 1 else "lnm2b"
p = pd.read_parquet(INTERIM / "panel.parquet")
d = implied_equity(pd.read_parquet(OUT / f"preds_{t}.parquet"), p, t)
d = d.merge(p[["permno", "date", "comnam", "m2b", "v2a", "v2s"]], on=["permno", "date"])
d["abs_e"] = d["eps"].abs()
cols = ["permno", "comnam", "date", "me", "be", "at", "sale", "debt", "m2b", "v2a", "v2s", "M_hat", "eps"]
print("|ε| quantiles:\n", d["abs_e"].quantile([0.5, 0.9, 0.99, 0.999]).round(2))
print("\n20 largest errors:\n", d.nlargest(20, "abs_e")[cols].to_string())
print("\nFirms among the 200 largest errors:", d.nlargest(200, "abs_e")["permno"].nunique())

print("\n10 largest market values and predictions (check for 1,000x errors):")
print(d.nlargest(10, "me")[cols].to_string())
print(d.nlargest(10, "M_hat")[cols].to_string())
