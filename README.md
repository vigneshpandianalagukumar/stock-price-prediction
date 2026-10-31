# Regime-Adaptive Bayesian Stacking for Cross-Sectional Equity Forecasting

A leakage-proof machine learning pipeline that ranks NSE-listed equities by expected
sector-neutral return and attaches a calibrated uncertainty interval to every forecast.

Three base learners (Random Forest, XGBoost, LSTM) are combined by a Bayesian
meta-learner whose weights shift with market volatility, validated under purged
walk-forward cross-validation with an embargo gap.

---

## Results at a glance

Evaluated on a fully out-of-sample holdout of approximately two years across
231 NSE-listed equities.

Holdout: 2024-06-07 to 2026-07-03, 512 trading days, 231 names, 118,267 scored
observations. Selected meta-learner: nonlinear.

| Metric | Value | Target / Benchmark |
|---|---|---|
| Mean cross-sectional Information Coefficient | **0.0240** | 0.02–0.05 typical in literature |
| Information Ratio of the IC (ICIR) | 0.263 | — |
| Days with positive IC | 63.5% | 50% |
| 95% interval coverage (PICP) | **97.2%** | 95% nominal |
| Long–short decile total return | **19.33%** | 8.12% benchmark |
| Long–short decile Sharpe | **0.899** | 0.346 benchmark |
| Long–short maximum drawdown | **−6.55%** | −18.33% benchmark |
| Long-only decile total return | 10.80% | 8.12% benchmark |
| Independent cycles at 20-day horizon | 129.2 | — |

All backtest figures are net of 10 basis points transaction cost per side.

**On the IC of 0.0240:** this is small, and that is the point. Individual equity
returns are dominated by noise, and the published cross-sectional literature reports
values in this range for signals that genuinely survive out-of-sample testing. Under
this validation scheme a substantially higher figure would indicate data leakage
rather than a better model.

**The ensemble did not beat its own best base learner.** On the holdout, Random
Forest alone scored IC 0.0453 against the stacked ensemble's 0.0240. The full table:

| Model | Holdout IC | ICIR |
|---|---|---|
| Random Forest | 0.0453 | 0.479 |
| XGBoost | 0.0363 | 0.421 |
| Linear meta-learner | 0.0299 | 0.302 |
| Nonlinear meta-learner *(selected)* | 0.0240 | 0.263 |
| LSTM | 0.0106 | 0.115 |

Model selection was made on internal validation without touching the holdout, which
is the correct protocol — the nonlinear meta-learner won there (0.0176 vs 0.0120) and
lost here. Reporting the holdout result of the model picked beforehand, rather than
the best number available after the fact, is the whole point of holding data back.
The honest reading is that the LSTM contributes little at this horizon and drags the
blend below the trees it is averaged with.

---

## The problem

Three structural weaknesses recur in applied machine learning for equity forecasting:

| Weakness | Consequence |
|---|---|
| Randomised k-fold cross-validation | Future information leaks into training; backtests collapse live |
| Raw return as prediction target | Model learns market direction (beta), not stock selection skill (alpha) |
| Point forecasts with no uncertainty | No basis for position sizing or risk limits |

A fourth issue is specific to Indian markets: standard split and dividend adjustment
does not correct for demergers and spin-offs, which produce single-day price falls of
30% or more that are structural rather than informational.

---

## Approach

**Leakage-proof validation.** Purged walk-forward cross-validation with an embargo gap
equal to the 20-day prediction horizon. Because a training label is computed from
prices extending 20 days forward, a conventional chronological split still leaves a
20-day window where training labels overlap test features. The embargo severs it.

**Sector-neutral cross-sectional target.** The contemporaneous sector mean forward
return is subtracted from each stock, and the residual is converted to a percentile
rank in [−0.5, +0.5]. This removes both market beta and sector beta, forcing the model
to identify relative outperformance within each sector.

**Heterogeneous feature assignment.** Each base learner receives a deliberately
different view of the feature space, so the ensemble achieves evidence diversity
rather than architectural diversity alone.

| Base learner | Features | Rationale |
|---|---|---|
| Random Forest | 14 cross-sectional rank features | Bagging is robust to noise; structural anchor |
| XGBoost | Those 14 + 11 five-day rank-momentum deltas (25 total) | Captures short-horizon interaction and acceleration |
| LSTM | 12-feature tensor over a 20-day lookback | Temporal trajectory without volume noise |

**Regime-aware Bayesian meta-learner.** Combination weights are formed as a function
of the India VIX z-score, so model influence varies continuously with market
conditions. The Bayesian formulation supplies posterior variance natively, giving a
95% prediction interval without a separate quantile estimation stage.

**Out-of-fold meta-training.** The meta-learner is fitted only on predictions the base
learners made for data they never saw during their own training.

---

## Architecture

```
Market Data (231 NSE tickers, India VIX, Nifty 50)
                  │
                  ▼
      Corporate Action Guard  ── structural breaks >35% neutralised
                  │
                  ▼
      Stationary Feature Engineering  ── cross-sectional percentile ranks
                  │
                  ▼
      Sector-Neutral Rank Target  ── FWD_RET − sector mean → rank [−0.5, +0.5]
                  │
                  ▼
      Purged Walk-Forward CV  ── embargo gap = 20-day horizon
                  │
     ┌────────────┴────────────┬──────────────────┐
     ▼                         ▼                  ▼
  Random Forest            XGBoost              LSTM
  (14 features)          (25 features)   (12 × 20-day tensor)
     └────────────┬────────────┴──────────────────┘
                  │  out-of-fold predictions
                  ▼
      Level-1 Bayesian Meta-Learner  ── VIX interaction terms
                  │
                  ▼
  Cross-sectional rank · Expected return · 95% interval
```

---

## Results in detail

### Diagnostics

![Diagnostic panels](figures/diagnostic_plots_h20.png)

**Top-left — holdout equity curves.** The long–short portfolio (green) stays within a
much narrower band than either directional alternative, which is why its Sharpe ratio
more than doubles theirs despite a smaller absolute return.

**Top-right — daily Information Coefficient.** Individual days are near coin-flips,
with visible clusters of negative IC. What matters is the persistent positive bias:
the mean sits at 0.0240 across the full holdout, positive on 63.5% of days.

**Bottom-left — meta-weight vs volatility regime.** As volatility rises, Random Forest
weight climbs from roughly 0.08 to 0.42 while LSTM weight roughly halves. This is the
regime adaptation measured rather than asserted. *(Shown using the linear meta-learner
for interpretability; the nonlinear variant was deployed.)*

**Bottom-right — interval calibration.** 97.2% of realised outcomes fall inside the
declared 95% interval. Coverage is honest, but it over-covers: an interval that
should miss one time in twenty misses one in thirty-five, so it is wider than the
data requires.

### Feature attribution

![SHAP summary](figures/shap_summary_h20.png)

### Live inference

![Model picks](figures/todays_picks_h20.png)

Predictions for the most recent trading date, which by definition have no realised
outcome yet and therefore cannot be scored. Performance claims rest on the holdout
evaluation above.

---

## Known limitations

Stated explicitly rather than left to be discovered:

- **Per-stock risk differentiation is weak, and measurably so.** Interval width has a
  relative dispersion of 0.062 across the deployed nonlinear meta-learner — the
  widths vary by about 6% around a mean of 0.299, so nearly every stock receives
  effectively the same ±15% band. The linear alternative is worse: dispersion 0.000,
  an interval of identical width for every name in the universe. Aggregate
  calibration is sound; per-stock granularity is not. This is the highest-priority fix.
- **The stack underperforms Random Forest alone** on the holdout (IC 0.0240 vs
  0.0453). See the table above. The architecture is demonstrated; the gain from
  stacking at this horizon is not.
- **Interval over-coverage.** 97.2% against a 95% target means the bands are
  conservative, which costs precision in any position-sizing use.
- **Base learner error correlation.** Random Forest and XGBoost share most of their
  feature set, so their errors correlate. This weakens identification of the
  regime-dependent weights.
- **Survivorship bias.** The universe is a current constituent list applied
  retrospectively; delisted and failed companies are absent.
- **Simplified transaction costs.** A flat 10 bps per side, independent of position
  size and liquidity. A realistic model would scale impact with traded volume.
- **Long-only drawdown.** The long-only decile beat the benchmark on return and Sharpe
  but with a deeper maximum drawdown (−22.19% vs −18.33%).
- **Single market, single primary horizon.** Sufficient to demonstrate the method, not
  to establish general performance.

---

## Repository structure

```
.
├── src/
│   ├── config.py      Constants. Must match the values the models were trained on
│   ├── universe.py    The 231 tickers and their sector map
│   ├── features.py    Download, corporate-action guard, feature engineering
│   └── inference.py   Artefact loading and scoring
├── predict.py         Command-line inference
├── tests/             Feature-ordering guards and an end-to-end regression test
├── notebook/          Full training pipeline
├── app/               Streamlit console
├── models/            Where trained artefacts go — see models/README.md
├── results/           Predictions, backtest series, summary metrics
├── figures/           Diagnostic and attribution plots
├── docs/              Project report
├── requirements.txt
└── README.md
```

---

## Inference

The complete inference path is here — download, feature construction, scoring, CLI.
The trained weights are not: the Random Forest artefact alone exceeds what this
repository can host. `models/README.md` lists the ten files and the one notebook cell
that writes them.

```bash
git clone https://github.com/vigneshpandianalagukumar/stock-price-prediction.git
cd stock-price-prediction

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run notebook/ once to populate models/, then:
python predict.py --top 10              # best and worst 10 names today
python predict.py --ticker RELIANCE     # one name
python predict.py --all --out picks.csv # full ranking to CSV
python predict.py --all --offline       # reuse the cached download
```

First run downloads about eleven years of daily data for 231 tickers and takes a few
minutes; it caches to `data/raw_prices.csv` so `--offline` is instant afterwards.

**Why the whole universe downloads even for one ticker.** Every feature is a daily
cross-sectional percentile rank — `RSI14_R` is not a property of the stock, it is
that stock's standing among the other 230 on that date. Score one name in isolation
and every rank collapses to zero. This is a consequence of the sector-neutral design,
not an inefficiency.

**Why `--start` should be left alone.** `VIX_Z` is an expanding z-score anchored to
the start of the series. Download from 2020 instead of 2015 and the same calendar day
produces a different `VIX_Z`, the meta-learner receives an input distribution it never
saw, and predictions degrade with no error raised.

### Reading the output

```
Ticker        PredRank  PredRet  VaR95_Lo  VaR95_Hi
GLENMARK.NS    +0.0620   +0.96%    -12.1%    +16.0%
```

`PredRet` is **sector-neutral**: +0.96% means Glenmark is expected to beat the average
Pharma name by that much over 20 trading days, not that the share price rises 0.96%.
`PredRank` is the cross-sectional ordering, which is the signal the model is actually
scored on. The interval is wide relative to the point estimate — honestly so.

### Verifying the install

```bash
pytest -q
```

Twelve checks run without network or model files, most of them pinning feature column
order. This matters more than it looks: the tree models were fitted on NumPy arrays
rather than DataFrames, so they carry no `feature_names_in_` and scikit-learn cannot
detect a reordered column at predict time. It returns confident wrong numbers instead
of raising. Two further end-to-end tests activate once `models/`, a cached snapshot
and a reference prediction file are all present, and confirm the repository still
reproduces the notebook's output.

### Streamlit console

```bash
streamlit run app/app.py
```

Opens at `http://localhost:8501`. Search any ticker to see its predicted rank, expected
sector-neutral return and 95% interval, browse the diagnostics and backtest, or trigger
a fresh scoring run from the first tab.

## Reproducing the pipeline

Open `notebook/` in Jupyter, Kaggle or Colab and run all cells. The horizon is set in
the first cell; the notebook writes tagged outputs (`*_h20.csv`, `*_h20.joblib`) so
multiple horizons can coexist.

A full run takes roughly 10 minutes on a standard CPU instance. No GPU is required.

---

## Stack

Python · scikit-learn · XGBoost · Keras · pandas · NumPy · SHAP · Streamlit · yfinance

---

## References

1. López de Prado, M. *Advances in Financial Machine Learning.* Wiley, 2018.
2. Gu, S., Kelly, B. and Xiu, D. "Empirical Asset Pricing via Machine Learning."
   *The Review of Financial Studies* 33(5), 2020.
3. Wolpert, D. H. "Stacked Generalization." *Neural Networks* 5(2), 1992.
4. Yao, Y., Pirš, G., Vehtari, A. and Gelman, A. "Bayesian Hierarchical Stacking:
   Some Models Are (Somewhere) Useful." *Bayesian Analysis* 17(4), 2022.

---

## Authors

Sandhyaa K · Vignesh Pandian A
Department of Electronics and Communication Engineering
Chennai Institute of Technology

Project Based Learning, 2026. Guide: Ms. Harshanya S.

---

## Disclaimer

Academic project. Not investment advice. The model produces a statistical ranking
under quantified uncertainty and has not been validated for live capital allocation.
