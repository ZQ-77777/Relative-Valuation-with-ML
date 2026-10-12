"""Run the full pipeline. Usage:
    python run_all.py --step build      # steps 2-7: clean, merge, features, targets, filters -> data/interim/panel.parquet
    python run_all.py --step train      # step 8: train the three ML models -> output/preds_*.parquet, shap_*.parquet
    python run_all.py --step evaluate   # steps 9-11: Tables 1/2/3/7/8 -> output/*.csv
    python run_all.py --step peers --date 2019-12-31 --permno 13511   # step 12: peers and SHAP for one firm
(permno 13511 is only a placeholder; look up the permno of your firm in CRSP)
"""
import argparse
import pandas as pd
from rv.config import RAW, INTERIM, OUT, TARGETS
from rv import crsp, compustat, link, industry, beta, features, targets, train
from rv import evaluate, benchmarks, portfolio, shap_tools, peers


def step_build():
    INTERIM.mkdir(parents=True, exist_ok=True)
    msf = pd.read_parquet(RAW / "crsp_msf.parquet")
    names = pd.read_parquet(RAW / "crsp_msenames.parquet")
    delist = pd.read_parquet(RAW / "crsp_msedelist.parquet")
    c = crsp.build_crsp_monthly(msf, names, delist)
    q = compustat.build_comp_quarterly(pd.read_parquet(RAW / "comp_fundq.parquet"))
    p = link.merge_crsp_comp(c, q, pd.read_parquet(RAW / "ccm_link.parquet"))
    ff = beta.load_ff_factors(RAW / "F-F_Research_Data_5_Factors_2x3.csv",
                              RAW / "F-F_Momentum_Factor.csv")
    p = p.merge(beta.capm_beta(c, ff), on=["permno", "date"], how="left")
    tab = industry.parse_siccodes((RAW / "Siccodes49.txt").read_text())
    p["ff49"] = industry.sic_to_ff49(p["siccd"], tab)
    raw_n = (len(p), p["gvkey"].nunique())
    p1 = targets.add_targets(p)
    p2 = targets.size_filter(p1)
    print(targets.table1(raw_n, p1, p2))
    X = features.build_features(p2)
    p2.to_parquet(INTERIM / "panel.parquet")
    X.to_parquet(INTERIM / "features.parquet")
    ff.to_parquet(INTERIM / "ff.parquet")


def step_train(n_jobs):
    OUT.mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(INTERIM / "panel.parquet")
    X = pd.read_parquet(INTERIM / "features.parquet")
    for t in TARGETS:
        pr, sh = train.run_all(p, X, t, n_jobs=n_jobs)
        pr.to_parquet(OUT / f"preds_{t}.parquet")
        sh.to_parquet(OUT / f"shap_{t}.parquet")


def step_evaluate():
    p = pd.read_parquet(INTERIM / "panel.parquet")
    ff = pd.read_parquet(INTERIM / "ff.parquet")
    res, t3 = {}, {}
    for t in TARGETS:
        pr = pd.read_parquet(OUT / f"preds_{t}.parquet")
        d = evaluate.implied_equity(pr, p, t)
        res["ML_" + t] = d
        ind = benchmarks.industry_harmonic(p, pr, t)
        res["IND_" + t[2:]] = evaluate.implied_equity(ind, p, t)
        d = d.merge(p[["permno", "date", "exchcd", "ret_next"]], on=["permno", "date"])
        t3["ML_" + t] = portfolio.factor_test(portfolio.long_short(d), ff)
        sh = pd.read_parquet(OUT / f"shap_{t}.parquet")
        shap_tools.variable_importance(sh).to_csv(OUT / f"table7_{t}.csv")
        shap_tools.category_importance(sh).to_csv(OUT / f"table8_{t}.csv")
    evaluate.table2(res).to_csv(OUT / "table2.csv")
    pd.concat(t3, axis=1).to_csv(OUT / "table3.csv")          # estimates and t-stats
    print(evaluate.table2(res))


def step_peers(date, permno, target="lnm2b"):
    p = pd.read_parquet(INTERIM / "panel.parquet")
    X = pd.read_parquet(INTERIM / "features.parquet")
    m = p["date"] == pd.Timestamp(date)
    names = p.loc[m].set_index("permno")["comnam"]
    pe, sh, summ = peers.firm_report(p[m], X[m], target, permno, names)
    print(summ)
    print("10 most comparable peers:\n", pe.head(10))
    print("10 least comparable peers:\n", pe.tail(10))
    print("Top 10 SHAP contributions:\n", sh.head(10))
    pe.to_csv(OUT / f"peers_{permno}_{date}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["build", "train", "evaluate", "peers"])
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--date")
    ap.add_argument("--permno", type=int)
    a = ap.parse_args()
    {"build": step_build, "train": lambda: step_train(a.n_jobs),
     "evaluate": step_evaluate, "peers": lambda: step_peers(a.date, a.permno)}[a.step]()
