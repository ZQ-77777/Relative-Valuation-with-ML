# Free-data version (without WRDS)

The free-data version builds `panel.parquet` and `features.parquet` in exactly the format of the WRDS version, from SEC EDGAR XBRL filings and public prices. Training, evaluation and peer-weight code run unchanged. With WRDS access, switch back to `python run_all.py --step build`.

## Data sources

| Content | Source | Code |
|---|---|---|
| Financial statements | SEC `companyfacts.zip` (XBRL; fairly complete from about 2009) | `rv/free/sec.py`, `rv/free/xbrl.py` |
| Tickers and exchanges | SEC `company_tickers_exchange.json` | `sec.download_tickers` |
| SIC codes | SEC submissions API (one request per CIK) | `sec.download_sic` |
| Daily prices and splits | yfinance (unofficial) | `rv/free/prices.py` |
| Factors, FF49 definitions | Kenneth French Data Library | `rv/free/french.py` (automatic download) |

The SEC requires a User-Agent with your name and email on every request: set `SEC_USER_AGENT` in `rv/config.py` first.

## Running

```bash
pip install -r requirements.txt
python -m tests.test_free                          # synthetic data, no network needed
python run_free.py --step download --limit 200     # trial run on 200 firms
python run_free.py --step download                 # full run (first time ~1-2 hours; resumable)
python run_free.py --step build
python run_all.py --step train --n_jobs 4
python run_all.py --step evaluate
```

If Yahoo rate-limits price downloads, wait 15-30 minutes and rerun the download step; only missing tickers are fetched.

## Key processing steps

- **Tag mapping.** Each Compustat item maps to candidate us-gaap tags in priority order, e.g. `sale` <- `Revenues`, then `RevenueFromContractWithCustomerExcludingAssessedTax`, ... (`xbrl.STOCK_TAGS`, `xbrl.FLOW_TAGS`).
- **Point-in-time values.** The same number reappears as a comparative in later filings and may be restated (10-K/A). Only the first-filed value is used, so restatements cause no look-ahead bias.
- **Trailing twelve months.** Annual periods use the 12-month value; interim periods use `current YTD + prior fiscal year - prior-year YTD`, which works for income statements and for YTD-only cash-flow statements.
- **Availability.** By default a period is usable from the month-end of its 10-Q/10-K filing date (`FREE_AVAIL = "filed"`); `"lag3"` reproduces the paper's 3-month lag.
- **Units.** XBRL amounts are in dollars and are converted to $ millions to match market equity.
- **Absent means zero.** XBRL omits zero items (a firm without short-term debt has no short-term-debt tag) where Compustat records 0. When total assets are reported, `dltt`, `dlc`, `pstk`, `mib`, `invt`, `xrd` and `dv` are set to 0 if missing.
- **Definitional alignment.** Missing `cogs` = sales - gross profit; SG&A reported in parts is summed and R&D added (Compustat XSGA includes R&D); missing operating income = sale - cogs - xsga - dp, else pretax income + interest - nonoperating income; `ceq = seq - pstk`; missing `lt = at - seq - mib`; `oibdp = oiadp + dp`; `icapt ~ dltt + ceq + pstk + mib`. `spi` has no tag and stays missing.
- **Market value and splits.** yfinance closes are adjusted for later splits, so SEC share counts are restated to the same basis using the product of split ratios after the share-count date.

## Free-data cleaning rules

| Rule | Typical effect | Reason |
|---|---|---|
| Cover-page and balance-sheet share counts cross-checked; counts more than 20x the firm's median corrected (if ~1,000x) or dropped | a few thousand filings | placeholder counts (e.g. 100 shares before a spin-off), counts reported in thousands |
| Firm-months with market value below $1 million dropped | ~0.1% | residual share-count errors |
| m2b outside [0.01, 1000], v2a outside [0.01, 100] or v2s outside [0.01, 1000] dropped | ~2% | reverse splits missing from the free price source |
| Periods whose total assets are 100x away from the firm's median dropped | small | unit errors in filings |

The paper drops no extreme multiples. These rules target free-data errors only, but they also remove a few genuinely extreme firms, so mean errors look slightly better than they would otherwise; medians are barely affected.

## Limitations

1. **Survivorship bias.** Ticker lists and prices cover currently listed firms only, so return tests are unreliable.
2. **Short sample.** About 2012-2024, versus 1980-2019 in the paper.
3. **Inconsistent tags.** Some variables (notably cost of goods sold) have lower coverage than in Compustat.
4. **Price quality.** yfinance is unofficial; some prices or split records are wrong.
5. **Identifiers.** CIK replaces permno and gvkey; each firm keeps one primary ticker.

The numbers are therefore not directly comparable with the paper, but the qualitative findings can be tested: whether GBM errors are below those of industry multiples, whether profitability, growth and efficiency remain the leading SHAP categories, and whether peer weights pick sensible comparables.

## Tests

`tests/test_free.py` uses synthetic companyfacts from `tests/fake_sec.py`, including repeated comparatives, a 10-K/A restatement, a revenue-tag change in 2018, a 2:1 split, a placeholder cover-page share count and share counts reported in thousands. It checks that TTM sales equal the true four-quarter sums, that market values equal the truth (including the split firm), and that the full pipeline produces the 97 features and trains.
