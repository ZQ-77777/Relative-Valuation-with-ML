"""Run the whole pipeline on synthetic data: python -m tests.test_pipeline
Uses real lightgbm if installed, otherwise an sklearn stand-in."""
try:
    import lightgbm  # noqa: F401
except ImportError:
    import tests.mock_lightgbm  # noqa: F401
import numpy as np
import pandas as pd
from tests.fake_data import make_raw, SIC_TEXT
from rv import crsp, compustat, link, industry, beta, features, targets, train
from rv import evaluate, benchmarks, portfolio, shap_tools, peers

msf, names, delist, lk, fq, ff = make_raw()
c = crsp.build_crsp_monthly(msf, names, delist)
q = compustat.build_comp_quarterly(fq)
p = link.merge_crsp_comp(c, q, lk)
p = p.merge(beta.capm_beta(c, ff), on=["permno", "date"], how="left")
tab = industry.parse_siccodes(SIC_TEXT)
p["ff49"] = industry.sic_to_ff49(p["siccd"], tab)
raw_n = (len(p), p["gvkey"].nunique())
p1 = targets.add_targets(p)
p2 = targets.size_filter(p1)
print(targets.table1(raw_n, p1, p2))
X = features.build_features(p2)
print("Feature matrix:", X.shape, "| five highest missing rates:\n", X.isna().mean().sort_values().tail(5))
months = p2["date"].unique()[:6]
sub = p2[p2["date"].isin(months)]
pr, sh = train.run_all(sub, X.loc[sub.index], "lnm2b")
d = evaluate.implied_equity(pr, p2, "lnm2b")
ind = evaluate.implied_equity(benchmarks.industry_harmonic(p2, pr, "lnm2b"), p2, "lnm2b")
print(evaluate.table2({"ML_lnm2b": d, "IND_m2b": ind}))
d2 = d.merge(p2[["permno", "date", "exchcd", "ret_next"]], on=["permno", "date"])
print(portfolio.factor_test(portfolio.long_short(d2), ff))
print(shap_tools.variable_importance(sh, 5))
print(shap_tools.category_importance(sh).head(3))
m = p2["date"] == months[0]
pe, s, summ = peers.firm_report(p2[m], X[m], "lnm2b", int(p2.loc[m, "permno"].iloc[0]),
                                p2[m].set_index("permno")["comnam"])
print(summ)
assert abs(summ["pred"] - summ["weighted_avg"]) < 1e-8
assert abs(pe["weight"].sum() - 1) < 1e-8
print(pe.head(3))

# application: treat one sample firm as a "new" firm
new = X[m].iloc[[5]]
res = peers.value_new_firm(p2[m], X[m], "lnm2b", new, denom=float(p2[m]["be"].iloc[5]))
print("New firm: predicted multiple", round(res["multiple"], 3), "implied equity value", round(res["equity_value"], 1))
print(res["peers"].head(3))
print("All tests passed")
