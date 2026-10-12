"""Step 5b: CAPM beta from 60-month rolling regressions of excess returns on market excess returns."""
import pandas as pd


def _read_french_csv(path) -> pd.DataFrame:
    """Read a Kenneth French CSV: skip the header notes and keep the monthly part (rows dated YYYYMM).
    Files differ in the number of note lines, so locate the header row (it starts with a comma)."""
    import io
    lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
    h = next(i for i, l in enumerate(lines) if l.startswith(",") and any(ch.isalpha() for ch in l))
    raw = pd.read_csv(io.StringIO("\n".join(lines[h:])), dtype=str, skip_blank_lines=False)
    raw = raw.rename(columns={raw.columns[0]: "ym"})
    raw["ym"] = raw["ym"].astype(str).str.strip()
    raw = raw[raw["ym"].str.fullmatch(r"\d{6}", na=False)]           # monthly rows only
    raw.columns = [c.strip().lower().replace("-", "") for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["ym"], format="%Y%m") + pd.offsets.MonthEnd(0)
    num = raw.drop(columns=["ym", "date"]).apply(pd.to_numeric, errors="coerce") / 100
    return pd.concat([raw[["date"]], num], axis=1).reset_index(drop=True)


def load_ff_factors(path_5f, path_mom) -> pd.DataFrame:
    """Read the monthly five-factor and momentum CSVs (values in percent).
    Output: date (month end), mktrf, smb, hml, rmw, cma, umd, rf (decimals)."""
    f5 = _read_french_csv(path_5f)
    mom = _read_french_csv(path_mom)
    mom = mom.rename(columns={c: "umd" for c in mom.columns if c.startswith("mom")})
    return f5.merge(mom[["date", "umd"]], on="date", how="inner")


def capm_beta(crsp: pd.DataFrame, ff: pd.DataFrame, window: int = 60,
              min_obs: int = 24) -> pd.DataFrame:
    """Inputs: crsp (permno, date, ret), ff (date, mktrf, rf).
    Output: permno, date, CAPMbeta (uses returns up to the current month only).
    min_obs=24 is an assumption; the paper states only "60 months"."""
    d = crsp[["permno", "date", "ret"]].merge(ff[["date", "mktrf", "rf"]], on="date")
    d = d.sort_values(["permno", "date"]).reset_index(drop=True)
    d["y"] = d["ret"] - d["rf"]
    d["x"] = d["mktrf"].where(d["y"].notna())      # only months with a non-missing return
    d["xy"], d["xx"] = d["x"] * d["y"], d["x"] ** 2
    g = d.groupby("permno")
    roll = lambda c: g[c].rolling(window, min_periods=min_obs).mean().reset_index(level=0, drop=True)
    mx, my, mxy, mxx = roll("x"), roll("y"), roll("xy"), roll("xx")
    d["CAPMbeta"] = (mxy - mx * my) / (mxx - mx ** 2)
    return d[["permno", "date", "CAPMbeta"]]
