"""Step 9b: traditional benchmark - industry harmonic-mean multiples (IND; Liu, Nissim and Thomas 2002).
Uses exactly the same blocks as the ML models: test firms get the harmonic mean of same-industry firms in the other four blocks."""
import numpy as np
import pandas as pd


def industry_harmonic(panel: pd.DataFrame, blocks: pd.DataFrame, target: str,
                      min_peers: int = 5) -> pd.DataFrame:
    """
    Inputs  panel  : permno, date, ff49 and the target column (e.g. lnm2b)
            blocks : permno, date, block from the ML predictions (identical splits)
    Output  DataFrame in the run_all format (pred = log of the harmonic-mean multiple)
    Falls back to the market-wide harmonic mean when an industry has fewer than min_peers training firms (assumption).
    """
    mult = target[2:]                       # 'lnm2b' → 'm2b'
    d = panel[["permno", "date", "ff49", "me", target, mult]].merge(
        blocks[["permno", "date", "block"]], on=["permno", "date"])
    out = []
    for date, dm in d.groupby("date"):
        for b in sorted(dm["block"].unique()):
            te, tr = dm[dm["block"] == b], dm[dm["block"] != b]
            hm = tr.groupby("ff49")[mult].agg(lambda s: 1 / np.mean(1 / s))
            cnt = tr.groupby("ff49")[mult].size()
            hm = hm[cnt >= min_peers]
            allm = 1 / np.mean(1 / tr[mult])
            p = te["ff49"].map(hm).fillna(allm)
            out.append(pd.DataFrame({"permno": te["permno"], "date": date, "block": b,
                                     "y": te[target], "pred": np.log(p),
                                     "ybar_train": tr[target].mean(),
                                     "mebar_train": tr["me"].mean()}))
    return pd.concat(out, ignore_index=True)
