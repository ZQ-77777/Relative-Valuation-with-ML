# Relative Valuation with Machine Learning — Replication

A Python replication of Geertsema, P. and Lu, H. (2023), "Relative Valuation with Machine Learning," *Journal of Accounting Research* 61(1): 329–376.

The pipeline runs end to end on two data sources:

- **CRSP/Compustat via WRDS**, following the paper (1980–2019), and
- **free SEC XBRL filings plus public prices** (2012–2024), for use without WRDS access.

Both produce the same panel format, so training, evaluation, SHAP analysis and peer weights run unchanged on either source.

## Results (free-data version)

2,599 NYSE/Nasdaq firms, January 2012 – December 2024, 245,898 firm-months.

| Model | med(\|ε\|) | med(ε) | R²oos (log multiple) | R²oos(M) | ρ(M, M̂) |
|---|---|---|---|---|---|
| GBM, m2b | 0.32 | 0.01 | 0.52 | 0.38 | 0.63 |
| Industry multiple, m2b | 0.43 | −0.20 | 0.03 | 0.17 | 0.48 |
| GBM, v2a | 0.32 | 0.01 | 0.75 | 0.36 | 0.61 |
| Industry multiple, v2a | 0.51 | −0.27 | 0.41 | 0.17 | 0.44 |
| GBM, v2s | 0.34 | 0.00 | 0.69 | 0.35 | 0.60 |
| Industry multiple, v2s | 0.59 | −0.35 | 0.15 | 0.20 | 0.54 |
| Paper, GBM m2b | 0.29 | −0.00 | 0.54 | 0.78 | 0.90 |

ε = M̂/M − 1. Industry multiple = industry harmonic mean (Liu, Nissim and Thomas 2002). R²oos is measured against the training-sample mean.

| Paper's claim | Free-data result |
|---|---|
| GBMs value firms more accurately than traditional models | Replicated (median error 26–42% lower) |
| GBM valuations are nearly unbiased; industry multiples undervalue | Replicated |
| Value drivers are congruent with DCF theory | Replicated |
| Industry is the single most important variable | Replicated (1st for m2b and v2a, 2nd for v2s) |
| Peer weights identify sensible comparables | Replicated (Moderna, Dec 2019: 1.351 vs. 1.36 in the paper) |
| High dollar-value accuracy | Partly replicated |
| Valuation errors predict next-month returns | Not replicated (likely survivorship bias, short sample, price-data noise) |

## Repository layout

```
rv/                    core package
  config.py            paths, sample period, model settings
  download.py          WRDS queries (CRSP, Compustat, CCM link table, comp.company)
  crsp.py              monthly stock file: market value, returns, delisting adjustment
  compustat.py         quarterly fundamentals: YTD-to-quarter, TTM, book equity, lags
  link.py              CRSP–Compustat link and point-in-time alignment
  industry.py, beta.py Fama–French 49 industries; 60-month CAPM beta
  targets.py           log m2b, v2a, v2s; size filters (paper Table 1)
  features.py          the 97 input variables (paper Table A.1)
  train.py             monthly 5-block out-of-sample LightGBM training with TreeSHAP
  evaluate.py          valuation errors and Table 2 metrics
  benchmarks.py        industry harmonic-mean benchmark
  portfolio.py         long–short portfolios, Fama–French six-factor regressions (Table 3)
  shap_tools.py        variable and category importance (Tables 7–9)
  peers.py             GBM peer weights (Appendix A.3), firm reports, valuing a new firm
  compare.py           WRDS vs. free-data comparison on common firm-months
  free/                free-data layer: SEC XBRL, prices, French data, panel builder
run_all.py             build / train / evaluate / peers (WRDS version, or after run_free.py)
run_free.py            download and build the free-data panel
check_outliers.py      list the most extreme valuation errors
tests/                 synthetic-data end-to-end tests and the peer-weight identity check
FREE_DATA.md           details of the free-data version: sources, processing, cleaning rules, limitations
```

## Quick start

Requires Python 3.10+.

```bash
pip install -r requirements.txt

# 1. Tests on synthetic data (no data access needed)
python -m tests.test_pipeline
python -m tests.test_free
python -m tests.validate_peer_weights

# 2a. Free-data version (set SEC_USER_AGENT in rv/config.py first)
python run_free.py --step download       # resumable; add --limit 200 for a trial run
python run_free.py --step build

# 2b. WRDS version
python -c "from rv.download import download_all; download_all('<wrds_username>')"
# place F-F_Research_Data_5_Factors_2x3.csv, F-F_Momentum_Factor.csv and Siccodes49.txt in data/raw/
python run_all.py --step build

# 3. Same for both
python run_all.py --step train --n_jobs 4
python run_all.py --step evaluate
python run_all.py --step peers --date 2019-12-31 --permno <PERMNO, or CIK in the free version>
```

Outputs are written to `output/` (`table2.csv`, `table3.csv`, `table7_*.csv`, `table8_*.csv`, `peers_*.csv`).

## Implementation choices not specified in the paper

| Item | Choice |
|---|---|
| Lagged values `l_x` in Table A.1 | Same fiscal quarter of the prior year (4-quarter lag) |
| Enterprise value | Market equity + book debt (dltt + dlc + pstk) |
| Early stopping | Up to 5,000 trees; stop after 50 rounds without validation improvement |
| Validation block | Block b + 1 when block b is the test set |
| CAPM beta | 60-month rolling window, minimum 24 months |
| Variable categories | Taken from the paper where reported, otherwise from the WRDS ratio-suite classification |
| Return tests | Fama–French six factors only; the paper's seven anomaly-cluster factors are not public |
| Industry benchmark rmse(ŷ) | Computed on log multiples, for comparability with the GBM |

Free-data version only:

| Item | Choice |
|---|---|
| Information lag | Actual SEC filing date (`FREE_AVAIL = "filed"`); `"lag3"` reproduces the paper's 3-month lag |
| Restatements | The first-filed value of each item is used (point-in-time) |
| Share counts | Cover-page and balance-sheet counts cross-checked; counts more than 20× the firm's median corrected (if ~1,000×) or dropped |
| Implausible multiples | m2b outside [0.01, 1000], v2a outside [0.01, 100] or v2s outside [0.01, 1000] dropped (2.14% of firm-months) |

## Tests

The tests run without any data access. If `lightgbm` is not installed, the end-to-end tests fall back to an sklearn stand-in (`tests/mock_lightgbm.py`); `tests/validate_peer_weights.py` needs the real `lightgbm`. The free-data tests use synthetic SEC filings that include repeated comparatives, a restatement, a revenue-tag change, a stock split and erroneous share counts.

## Data

No data are included in this repository. CRSP and Compustat are licensed through WRDS and must not be redistributed. SEC filings, public prices and Kenneth French's factor data are downloaded by the scripts.

## Acknowledgments

The peer-weight algorithm follows Appendix A.3 of the paper and the authors' public example code (`pgeertsema/peerweights`). I also drew on Xie Bocheng's replication notes.

## Citation

Geertsema, P., and H. Lu. 2023. Relative valuation with machine learning. *Journal of Accounting Research* 61 (1): 329–376. https://doi.org/10.1111/1475-679X.12464

## License

Code is released under the MIT License (see `LICENSE`). Data obtained through WRDS remain subject to their own license terms.
