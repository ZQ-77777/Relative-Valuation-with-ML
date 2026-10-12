"""End-to-end test of the free-data version: python -m tests.test_free"""
try:
    import lightgbm  # noqa: F401
except ImportError:
    import tests.mock_lightgbm  # noqa: F401  (sklearn stand-in)
import numpy as np
import pandas as pd
import rv.config as cfg
from tests.fake_sec import make_sec
from tests.fake_data import SIC_TEXT
from rv.free import xbrl, prices, sec, build
from rv import train, evaluate


docs, truth, daily, tk_raw, sic = make_sec()
facts = pd.concat([xbrl.facts_long(d) for d in docs], ignore_index=True)
print("XBRL facts:", len(facts))

# 1) TTM must equal the true sum of the last four quarters
q = xbrl.build_quarterly(facts)
t = truth.sort_values(["cik", "datadate"]).copy()
t["ttm_true"] = t.groupby("cik")["sale_q"].transform(lambda s: s.rolling(4).sum())
t["gvkey"] = t["cik"].astype(str).str.zfill(10)
chk = q.merge(t, on=["gvkey", "datadate"]).dropna(subset=["sale", "ttm_true"])
err = (chk["sale"] / chk["ttm_true"] - 1).abs().max()   # q is in $ millions, the same unit as the truth
print("Max relative TTM sales error:", err, "| periods checked:", len(chk))
assert err < 1e-9, "TTM is wrong (restated values used or tag change mishandled?)"
assert (q["avail"] - q["datadate"]).dt.days.min() >= 28, "availability date before period end"
f2 = chk[chk["cik"] == 1002]
sga_true = f2["ttm_true"] * (0.2 + 0.04)                 # SG&A parts summed + R&D (Compustat definition)
assert (f2["xsga"] / sga_true - 1).abs().max() < 1e-9, "summing SG&A parts failed"
assert (f2["dlc"] == 0).all(), "missing short-term debt should be set to 0"
assert q.loc[q["gvkey"] == "0000001003", "oiadp"].notna().mean() > 0.9, "operating income fill-in failed"
print("SG&A summation, zero fill-ins, operating income fill-in: passed")

# 2) market value must equal the truth before and after splits
tk = sec.save_tickers(tk_raw, save=False)
mo = prices.to_monthly(daily[["ticker", "date", "close", "adj_close", "split"]])
m = prices.add_market_cap(mo, xbrl.shares_table(facts), tk)
true_m = (daily.assign(date=daily["date"] + pd.offsets.MonthEnd(0))
          .groupby(["ticker", "date"])["me_true"].last().reset_index()
          .merge(tk[["cik", "ticker"]], on="ticker"))
cm = m.merge(true_m, on=["cik", "date"])
me_err = (cm["me"] / cm["me_true"] - 1).abs().max()
print("Max relative market-value error (incl. split firm):", me_err)
assert me_err < 1e-9

# 3) end to end: build the panel and features, train two months
ff = pd.DataFrame({"date": pd.date_range("2007-01-31", "2021-03-31", freq="ME")})
rng = np.random.default_rng(0)
for c in ["mktrf", "smb", "hml", "rmw", "cma", "umd"]:
    ff[c] = rng.normal(0.005, 0.04, len(ff))
ff["rf"] = 0.002
panel, X, t1 = build.build_free_panel(facts, daily[["ticker", "date", "close", "adj_close", "split"]],
                                      tk, sic, ff, SIC_TEXT, start="2012-01-31", end="2020-12-31", save=False)
print(t1)
med = panel[["m2b", "v2a", "v2s"]].median()
print("Median multiples:", med.round(2).to_dict())
assert med.between(0.05, 50).all(), "multiples out of range: check units of accounting data vs. market value"
print("Feature matrix:", X.shape, "| features entirely missing:", int(X.isna().all().sum()))
months = panel["date"].drop_duplicates().sort_values().iloc[-2:]
sub = panel[panel["date"].isin(months)]
pr, sh = train.run_all(sub, X.loc[sub.index], "lnm2b", min_firms=10)
print(evaluate.table2({"ML_lnm2b(free)": evaluate.implied_equity(pr, panel, "lnm2b")}))
print("All tests passed")
