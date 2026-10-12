"""Free data, step 3: month-end prices, monthly returns and market values.

Prices default to yfinance (an unofficial interface that changes occasionally and covers only current tickers).
Any other source works if reshaped into the daily_long format and passed to to_monthly.

Splits and market value: yfinance Close is split-adjusted backward for later splits, while SEC share
counts are as of their own dates. For consistency, historical share counts are restated to today's basis:
    adjusted shares = shares x product of all split ratios after the share-count date, F(sh_end)
    market value    = split-adjusted close x adjusted shares
"""
import numpy as np
import pandas as pd


def download_yf(tickers, start: str, end: str, batch: int = 50, pause: float = 3.0,
                max_wait: int = 3) -> pd.DataFrame:
    """
    Output daily_long: ticker, date, close (split-adjusted), adj_close (total return), split (ratio; 0 if none)
    Yahoo rate-limits frequent requests (YFRateLimitError). The function pauses `pause` seconds between batches;
    if almost nothing comes back for a batch, it is treated as rate-limited and retried after 60, 120 and 180 seconds.
    """
    import time
    import yfinance as yf
    out = []
    tickers = list(tickers)
    for i in range(0, len(tickers), batch):
        part = tickers[i:i + batch]
        for attempt in range(max_wait + 1):
            raw = yf.download(part, start=start, end=end, auto_adjust=False, actions=True,
                              group_by="ticker", progress=False, threads=True)
            got = []
            for t in part:
                try:
                    d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                except KeyError:
                    continue
                if d.empty or "Close" not in d:      # rate-limited requests may return an empty frame
                    continue
                d = d.dropna(subset=["Close"])
                if d.empty:
                    continue
                split = d["Stock Splits"] if "Stock Splits" in d else 0.0
                got.append(pd.DataFrame({"ticker": t, "date": d.index, "close": d["Close"].values,
                                         "adj_close": d["Adj Close"].values,
                                         "split": np.asarray(split, dtype=float)}))
            if len(got) >= 0.3 * len(part) or attempt == max_wait:
                break
            wait = 60 * (attempt + 1)
            print(f"Possibly rate-limited (got {len(got)}/{len(part)} tickers); retrying in {wait} seconds")
            time.sleep(wait)
        out.extend(got)
        print(f"Prices: {min(i + batch, len(tickers))}/{len(tickers)} done")
        time.sleep(pause)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["ticker", "date", "close", "adj_close", "split"])


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Input   daily_long (see above)
    Output  ticker, date (month end), close, ret (monthly total return), ret_next, F (product of later splits)
    """
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
    d = d.sort_values(["ticker", "date"])
    d["split"] = d["split"].fillna(0).replace(0, 1.0)
    # F(t) = product of split ratios after date t (reverse cumulative product, shifted by one day)
    d["F"] = (d.iloc[::-1].groupby("ticker")["split"].cumprod().iloc[::-1]
              / d["split"])
    d["month"] = d["date"] + pd.offsets.MonthEnd(0)
    m = d.groupby(["ticker", "month"]).agg(close=("close", "last"),
                                            adj=("adj_close", "last"),
                                            F=("F", "last")).reset_index()
    m = m.rename(columns={"month": "date"}).sort_values(["ticker", "date"])
    g = m.groupby("ticker")
    prev = g[["date", "adj"]].shift(1)
    ok = prev["date"] == m["date"] - pd.offsets.MonthEnd(1)
    m["ret"] = (m["adj"] / prev["adj"] - 1).where(ok)
    nxt = g[["date", "ret"]].shift(-1)
    m["ret_next"] = nxt["ret"].where(nxt["date"] == m["date"] + pd.offsets.MonthEnd(1))
    return m.drop(columns="adj").reset_index(drop=True)


def add_market_cap(monthly: pd.DataFrame, shares: pd.DataFrame, tickers: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs  monthly : output of to_monthly
            shares  : output of xbrl.shares_table (cik, sh_end, sh_filed, shares)
            tickers : output of sec.download_tickers (cik, ticker, name, exchcd)
    Output  a CRSP-like monthly table: permno (= CIK), date, me ($ millions), ret, ret_next, exchcd, shrcd, comnam
    Share counts are those filed by each month end (as-of merge on sh_filed).
    """
    m = monthly.merge(tickers[["cik", "ticker", "name", "exchcd"]], on="ticker", how="inner")
    sh = shares.copy()
    sh["sh_avail"] = sh["sh_filed"] + pd.offsets.MonthEnd(0)
    sh = sh.sort_values("sh_end").groupby(["cik", "sh_avail"]).tail(1)   # several filings in a month: latest period
    # split factor F(sh_end) for each share count: F of the stock in the month of sh_end
    fm = monthly[["ticker", "date", "F"]].merge(tickers[["cik", "ticker"]], on="ticker")
    sh["date_key"] = sh["sh_end"] + pd.offsets.MonthEnd(0)
    sh = sh.merge(fm.rename(columns={"date": "date_key", "F": "F_sh"})[["cik", "date_key", "F_sh"]],
                  on=["cik", "date_key"], how="left")
    sh["F_sh"] = sh["F_sh"].fillna(1.0)
    sh["shares_adj"] = sh["shares"] * sh["F_sh"]
    # unit errors: some filings report shares in thousands (or the reverse), ~1,000x away from the firm's median.
    # Correct ~1,000x first; counts still more than 20x away are dropped (the previous count carries forward)
    med = sh.groupby("cik")["shares_adj"].transform("median")
    r = sh["shares_adj"] / med
    sh.loc[r.between(1 / 3000, 1 / 300), "shares_adj"] *= 1000
    sh.loc[r.between(300, 3000), "shares_adj"] /= 1000
    r = sh["shares_adj"] / med
    n_bad = int(((r < 1 / 20) | (r > 20)).sum())
    if n_bad:
        print(f"Dropped {n_bad} filings with share counts more than 20x away from the firm's median")
    sh = sh[(r >= 1 / 20) & (r <= 20)]
    out = pd.merge_asof(m.sort_values("date"), sh.sort_values("sh_avail")[["cik", "sh_avail", "shares_adj"]],
                        left_on="date", right_on="sh_avail", by="cik", direction="backward",
                        tolerance=pd.Timedelta(days=200))
    out["me"] = out["close"] * out["shares_adj"] / 1e6
    out = out.dropna(subset=["me"])
    out = out.rename(columns={"name": "comnam"})
    out["permno"] = out["cik"].astype(int)
    out["shrcd"] = 10
    return out[["permno", "cik", "date", "me", "ret", "ret_next", "exchcd", "shrcd", "comnam"]] \
        .sort_values(["permno", "date"]).reset_index(drop=True)
