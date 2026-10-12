"""Step 5a: Fama-French 49 industries. Parses Siccodes49.txt from Kenneth French's website and maps SIC codes to 1-49."""
import re
import numpy as np
import pandas as pd


def parse_siccodes(text: str) -> pd.DataFrame:
    """Input: full text of Siccodes49.txt. Output: ff49, ff49_name, sic_lo, sic_hi."""
    rows, cur, name = [], None, None
    for line in text.splitlines():
        m = re.match(r"^\s*(\d{1,2})\s+(\S+)\s+", line)
        r = re.match(r"^\s+(\d{4})-(\d{4})", line)
        if r and cur is not None:
            rows.append((cur, name, int(r.group(1)), int(r.group(2))))
        elif m:
            cur, name = int(m.group(1)), m.group(2)
    return pd.DataFrame(rows, columns=["ff49", "ff49_name", "sic_lo", "sic_hi"])


def sic_to_ff49(sic: pd.Series, table: pd.DataFrame) -> pd.Series:
    """Input: SIC codes. Output: FF49 codes; SIC codes not listed map to 49 (Other)."""
    s = pd.to_numeric(sic, errors="coerce").to_numpy()
    out = np.full(len(s), 49, dtype=float)
    for row in table.itertuples():
        out[(s >= row.sic_lo) & (s <= row.sic_hi)] = row.ff49
    out[np.isnan(s)] = np.nan
    return pd.Series(out, index=sic.index)
