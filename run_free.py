"""Free-data version (for use without WRDS). Usage:
    1) set SEC_USER_AGENT in rv/config.py to your name and email
    2) pip install requests yfinance
    3) Kenneth French factor and industry files download automatically to data/raw_free/ (manual fallback is printed on failure)
    4) python run_free.py --step download     # SEC filings, tickers, SIC codes, prices (first run ~1-2 hours)
       python run_free.py --step build        # writes data/interim/panel.parquet and features.parquet
    5) python run_all.py --step train --n_jobs 4
       python run_all.py --step evaluate
       python run_all.py --step peers --date 2024-12-31 --permno <CIK>
"""
import argparse
import pandas as pd
from rv.config import FREE_RAW, FREE_PRICE_START, FREE_END
from rv import beta
from rv.free import sec, xbrl, prices, build, french


def _write_part(buf, fdir, k):
    part = pd.concat(buf, ignore_index=True)
    for c in ["tag", "unit", "form", "fp", "accn"]:
        part[c] = part[c].astype(str)
    part.to_parquet(fdir / f"part_{k:04d}.parquet", index=False)


def step_download(limit=None, force=False):
    """Completed parts are skipped; --force downloads everything again."""
    FREE_RAW.mkdir(parents=True, exist_ok=True)
    french.download_french(FREE_RAW)
    tp = FREE_RAW / "tickers.parquet"
    tk = pd.read_parquet(tp) if (tp.exists() and not force) else sec.download_tickers()
    if limit:
        tk = tk.head(limit)
    zp = FREE_RAW / "companyfacts.zip"
    if force or not zp.exists():
        print("Downloading companyfacts.zip ...")
        sec.download_companyfacts_bulk(zp)
    fdir = FREE_RAW / "facts"                     # stored in chunks to limit memory use
    done = set()
    if fdir.exists() and not force:
        done = set(pd.read_parquet(fdir, columns=["cik"])["cik"].unique())
    ciks = set(tk["cik"])
    todo = ciks - done
    if todo:
        fdir.mkdir(parents=True, exist_ok=True)
        start = len(list(fdir.glob("part_*.parquet")))
        buf, n_files, n_seen = [], start, 0
        for doc in sec.iter_companyfacts(zp, todo):
            f = xbrl.facts_long(doc)
            n_seen += 1
            if not f.empty:
                buf.append(f)
            if len(buf) >= 300:
                _write_part(buf, fdir, n_files); n_files += 1; buf = []
                print(f"Filings: {n_seen}/{len(todo)} firms processed")
        if buf:
            _write_part(buf, fdir, n_files)
        print(f"Filings: done, {n_seen} firms processed")
    sec.download_sic(sorted(ciks))                    # cached; fetches only missing firms
    pp = FREE_RAW / "prices_daily.parquet"
    have = set(pd.read_parquet(pp, columns=["ticker"])["ticker"]) if pp.exists() and not force else set()
    todo = [t for t in tk["ticker"] if t not in have]
    if todo:
        end = (pd.Timestamp(FREE_END) + pd.offsets.MonthEnd(2)).strftime("%Y-%m-%d")
        new = prices.download_yf(todo, FREE_PRICE_START, end)
        old = pd.read_parquet(pp) if have else None
        pd.concat([old, new], ignore_index=True).to_parquet(pp) if old is not None else new.to_parquet(pp)
    print("Download complete. Next: python run_free.py --step build")


def _read_facts():
    f = pd.read_parquet(FREE_RAW / "facts")
    for c in ["tag", "unit", "form", "fp", "accn"]:
        f[c] = f[c].astype("category")          # saves memory
    return f


def step_build():
    french.download_french(FREE_RAW)                  # fetches factor files if missing
    ff = beta.load_ff_factors(FREE_RAW / "F-F_Research_Data_5_Factors_2x3.csv",
                              FREE_RAW / "F-F_Momentum_Factor.csv")
    panel, X, t1 = build.build_free_panel(
        _read_facts(),
        pd.read_parquet(FREE_RAW / "prices_daily.parquet"),
        pd.read_parquet(FREE_RAW / "tickers.parquet"),
        pd.read_parquet(FREE_RAW / "sic.parquet"), ff,
        (FREE_RAW / "Siccodes49.txt").read_text(encoding="utf-8", errors="ignore"))
    print(t1)
    print("Feature missing rates (top 10):\n", X.isna().mean().sort_values().tail(10))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["download", "build"])
    ap.add_argument("--limit", type=int, help="use only the first N firms (trial run)")
    ap.add_argument("--force", action="store_true", help="ignore existing files and download everything again")
    a = ap.parse_args()
    step_download(a.limit, a.force) if a.step == "download" else step_build()
