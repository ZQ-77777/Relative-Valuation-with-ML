"""Step 2: CRSP monthly data -> one row per stock-month with market value and next-month return."""
import numpy as np
import pandas as pd


def build_crsp_monthly(msf: pd.DataFrame, names: pd.DataFrame,
                       delist: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Inputs
      msf    : permno, date, prc, shrout, ret        (crsp.msf)
      names  : permno, namedt, nameendt, shrcd, exchcd, siccd, comnam (crsp.msenames)
      delist : permno, dlstdt, dlret (optional; adjusts returns in the delisting month)
    Output (one row per permno-month)
      permno, date (month end), me (market equity, $ millions), ret, ret_next (next-month return),
      exchcd, shrcd, siccd, comnam
    """
    df = msf[["permno", "date", "prc", "shrout", "ret"]].copy()
    df["date"] = pd.to_datetime(df["date"])

    nm = names.copy()
    nm["nameendt"] = pd.to_datetime(nm["nameendt"]).fillna(pd.Timestamp("2099-12-31"))
    df = df.merge(nm, on="permno", how="left")
    # keep the name record valid on each date (share code and exchange change over time)
    df = df[(df["date"] >= df["namedt"]) & (df["date"] <= df["nameendt"])]

    # delisting month return = (1 + ret)(1 + dlret) - 1
    if delist is not None:
        dl = delist.copy()
        dl["date"] = pd.to_datetime(dl["dlstdt"]) + pd.offsets.MonthEnd(0)
        df["mdate"] = df["date"] + pd.offsets.MonthEnd(0)
        df = df.merge(dl[["permno", "date", "dlret"]].rename(columns={"date": "mdate"}),
                      on=["permno", "mdate"], how="left")
        r = df["ret"].fillna(0) if "ret" in df else 0
        df["ret"] = np.where(df["dlret"].notna(),
                             (1 + r) * (1 + df["dlret"]) - 1, df["ret"])
        df = df.drop(columns=["mdate", "dlret"])

    df["date"] = df["date"] + pd.offsets.MonthEnd(0)            # align to month end
    df = df[df["shrcd"].isin([10, 11]) & df["exchcd"].isin([1, 2, 3])]
    df["me"] = df["prc"].abs() * df["shrout"] / 1000             # thousands of shares x $ -> $ millions
    df.loc[df["me"] <= 0, "me"] = np.nan
    df = (df.sort_values(["permno", "date"])
            .drop_duplicates(["permno", "date"], keep="last"))

    # next-month return, only when the next row is the following calendar month
    nxt = df.groupby("permno")[["date", "ret"]].shift(-1)
    ok = nxt["date"] == df["date"] + pd.offsets.MonthEnd(1)
    df["ret_next"] = nxt["ret"].where(ok)
    return df[["permno", "date", "me", "ret", "ret_next", "exchcd", "shrcd",
               "siccd", "comnam"]].reset_index(drop=True)
