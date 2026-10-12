"""Free data, step 1: download filings (XBRL companyfacts), tickers and exchanges, and SIC codes from SEC EDGAR.

SEC fair-access rules: every request needs a User-Agent (name + email; see config.SEC_USER_AGENT)
and at most 10 requests per second. The bulk file (companyfacts.zip) is downloaded once.
If a URL below stops working, check the SEC page "EDGAR Application Programming Interfaces".
"""
import io
import json
import time
import zipfile
from pathlib import Path

import pandas as pd

from ..config import FREE_RAW, SEC_USER_AGENT

URL_TICKERS = "https://www.sec.gov/files/company_tickers_exchange.json"
URL_FACTS_BULK = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
URL_SUBMISSION = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

EXCH_CODE = {"NYSE": 1, "Nasdaq": 3}   # matches CRSP exchcd: 1 = NYSE, 3 = NASDAQ


def _get(url: str, stream: bool = False):
    import requests
    if "your_email" in SEC_USER_AGENT:
        raise ValueError("Set SEC_USER_AGENT in rv/config.py to your name and email first")
    for attempt in range(3):                       # up to three retries on network errors
        try:
            r = requests.get(url, headers={"User-Agent": SEC_USER_AGENT},
                             timeout=(10, 60), stream=stream)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == 2:
                raise
            print(f"Request failed, retrying in 5 seconds ({e})")
            time.sleep(5)


def download_tickers() -> pd.DataFrame:
    """
    Output  cik, name, ticker, exchange, exchcd - one row per CIK (primary ticker),
            NYSE and Nasdaq listings only. Saved as raw_free/tickers.parquet
    Note    this is the *current* listing: delisted firms are absent (a source of survivorship bias).
    """
    FREE_RAW.mkdir(parents=True, exist_ok=True)
    js = _get(URL_TICKERS).json()
    df = pd.DataFrame(js["data"], columns=js["fields"])
    return save_tickers(df)


def save_tickers(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Keep NYSE/Nasdaq, one primary ticker per CIK, and add exchcd."""
    df = df[df["exchange"].isin(EXCH_CODE)].copy()
    # a firm may have several tickers (share classes, preferred): prefer the shortest without "-"
    df["_rank"] = df["ticker"].str.contains("-").astype(int) * 100 + df["ticker"].str.len()
    df = df.sort_values(["cik", "_rank"]).drop_duplicates("cik").drop(columns="_rank")
    df["exchcd"] = df["exchange"].map(EXCH_CODE)
    if save:
        df.to_parquet(FREE_RAW / "tickers.parquet")
    return df.reset_index(drop=True)


def download_companyfacts_bulk(dest: Path | None = None) -> Path:
    """Download the nightly companyfacts.zip from the SEC (several GB uncompressed). Returns the local path."""
    dest = dest or FREE_RAW / "companyfacts.zip"
    FREE_RAW.mkdir(parents=True, exist_ok=True)
    with _get(URL_FACTS_BULK, stream=True) as r, open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return dest


def iter_companyfacts(zip_path: Path, ciks: set[int]):
    """Read CIK##########.json files from the zip one at a time, yielding only firms in ciks (saves memory)."""
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.startswith("CIK"):
                continue
            cik = int(name[3:13])
            if cik in ciks:
                try:
                    with z.open(name) as fh:
                        doc = json.load(io.TextIOWrapper(fh, encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue                       # skip corrupt or empty files
                if not isinstance(doc, dict):
                    continue
                doc["cik"] = cik                   # the CIK in the file name is authoritative (some files lack "cik")
                yield doc


def download_sic(ciks, cache: Path | None = None, sleep: float = 0.15) -> pd.DataFrame:
    """
    Fetch SIC codes from the submissions API, one CIK at a time; cached, so interrupted runs resume.
    Output  cik, sic, name
    """
    cache = cache or FREE_RAW / "sic.parquet"
    done = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["cik", "sic", "name"])
    todo = [c for c in ciks if c not in set(done["cik"])]
    rows = []
    for i, cik in enumerate(todo):
        try:
            js = _get(URL_SUBMISSION.format(cik=cik)).json()
            rows.append((cik, pd.to_numeric(js.get("sic"), errors="coerce"), js.get("name")))
        except Exception as e:                         # one failure does not stop the run
            print("Skipping", cik, e)
        time.sleep(sleep)                              # stay under 10 requests per second
        if (i + 1) % 200 == 0 or i + 1 == len(todo):
            new = pd.DataFrame(rows, columns=["cik", "sic", "name"])
            done = new if done.empty else pd.concat([done, new], ignore_index=True)
            done.to_parquet(cache)
            rows = []
            print(f"SIC: {i + 1}/{len(todo)} done")
    return done
