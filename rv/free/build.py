"""Free data, step 4: assemble panel.parquet / features.parquet in exactly the WRDS-version format.
Then run run_all.py --step train / evaluate / peers without changes."""
import pandas as pd

from .. import beta, features, industry, link, targets
from ..config import FREE_AVAIL, FREE_END, FREE_START, INTERIM
from . import prices, xbrl


def build_free_panel(facts: pd.DataFrame, daily: pd.DataFrame, tickers: pd.DataFrame,
                     sic: pd.DataFrame, ff: pd.DataFrame, sic_text: str,
                     start: str = FREE_START, end: str = FREE_END, save: bool = True):
    """
    Inputs  facts   : output of xbrl.facts_long (all firms)
            daily   : output of prices.download_yf (or any source reshaped to the same columns)
            tickers : output of sec.download_tickers
            sic     : output of sec.download_sic (cik, sic)
            ff      : output of beta.load_ff_factors
            sic_text: full text of Siccodes49.txt
    Output  (panel, X, table1); written to data/interim/ when save=True
    """
    q = xbrl.build_quarterly(facts, avail_mode=FREE_AVAIL)
    print("Coverage of base variables (closer to 1 is better):\n", xbrl.coverage(q).to_string())
    sh = xbrl.shares_table(facts)
    m = prices.add_market_cap(prices.to_monthly(daily), sh, tickers)
    m = m.merge(sic[["cik", "sic"]].rename(columns={"sic": "siccd"}), on="cik", how="left")
    bad = m["me"] < 1                      # market value under $1 million: almost always a share-count error
    print(f"Dropped {int(bad.sum())} firm-months with market value below $1 million ({bad.mean():.2%})")
    m = m[~bad]

    # an identity link table lets us reuse the WRDS merge function: gvkey = 10-digit CIK string
    lk = pd.DataFrame({"permno": m["permno"].unique()})
    lk["gvkey"] = lk["permno"].astype(str).str.zfill(10)
    lk["linkdt"], lk["linkenddt"] = pd.Timestamp("1900-01-01"), pd.NaT
    p = link.merge_crsp_comp(m.drop(columns="cik"), q, lk)

    p = p.merge(beta.capm_beta(m, ff), on=["permno", "date"], how="left")
    p["ff49"] = industry.sic_to_ff49(p["siccd"], industry.parse_siccodes(sic_text))
    fin = p["ff49"].between(45, 48).map({True: "financial (FF49 45-48)", False: "non-financial"})
    print("Coverage by sector (financial firms have no current/non-current split or cost of goods sold):\n",
          xbrl.coverage(p, by=fin).to_string())
    raw_n = (len(p), p["gvkey"].nunique())
    p1 = targets.add_targets(p)
    # last-resort cleaning for free data: economically impossible multiples come from price or share errors (mostly missing reverse splits)
    ok = (p1["m2b"].between(0.01, 1000) & p1["v2a"].between(0.01, 100)
          & p1["v2s"].between(0.01, 1000))
    print(f"Dropped {int((~ok).sum())} firm-months with implausible multiples ({(~ok).mean():.2%})")
    p1 = p1[ok]
    p2 = targets.size_filter(p1, start=start, end=end)
    t1 = targets.table1(raw_n, p1, p2)
    X = features.build_features(p2)
    if save:
        INTERIM.mkdir(parents=True, exist_ok=True)
        p2.to_parquet(INTERIM / "panel.parquet")
        X.to_parquet(INTERIM / "features.parquet")
        ff.to_parquet(INTERIM / "ff.parquet")
    return p2, X, t1
