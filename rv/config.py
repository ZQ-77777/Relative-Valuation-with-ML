"""Global settings: paths, sample period and model parameters. All other modules read from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"          # raw downloads (step 1)
INTERIM = ROOT / "data" / "interim"  # intermediate panel
OUT = ROOT / "output"                # predictions and tables

START = "1980-01-31"   # first valuation month (paper: January 1980)
END = "2019-12-31"     # last valuation month (paper: December 2019)
DATA_START = "1974-01-01"  # earlier raw data: one year for lags, five years for CAPM beta

INFO_LAG_MONTHS = 3    # accounting data available 3 months after fiscal period end (paper, p. 337)
SIZE_PCTL = 0.10       # drop firms below the monthly 10th percentile of assets, book equity or sales (Table 1)
TARGETS = ["lnm2b", "lnv2a", "lnv2s"]

N_BLOCKS = 5           # five random blocks per month: 1 test, 1 validation, 3 training -> 60:20:20
SEED = 2023

# LightGBM parameters: the paper tunes only the number of trees by early stopping; all else default (footnote 7)
LGB_PARAMS = dict(
    objective="regression",   # squared error (L2)
    learning_rate=0.1,        # λ
    num_leaves=31,
    min_data_in_leaf=20,
    verbose=-1,
    seed=SEED,
    deterministic=True,
    num_threads=1,            # parallelism is across months instead
)
MAX_TREES = 5000              # upper bound before early stopping (assumption; not stated in the paper)
EARLY_STOP = 50               # rounds without validation improvement before stopping (assumption)

# ---------- free-data version (rv/free/, for use without WRDS) ----------
FREE_RAW = ROOT / "data" / "raw_free"
FREE_START = "2012-01-31"     # XBRL coverage is fairly complete from ~2009; leave two years for lags and beta
FREE_END = "2024-12-31"
FREE_PRICE_START = "2007-01-01"   # five extra years of prices for the 60-month beta
# The SEC requires a User-Agent with a contact name and email on every request
SEC_USER_AGENT = "Your Name your_email@example.com"
FREE_AVAIL = "filed"          # "filed": available at the actual SEC filing date; "lag3": paper convention, period end + 3 months
