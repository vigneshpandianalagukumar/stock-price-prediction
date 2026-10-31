# Trained artefacts

**These files are not in the repository.** They are too large to host here, so this
folder ships empty and `predict.py` will stop with a clear message until the files
are present.

## What belongs here

| File | What it is |
|---|---|
| `rf_level0_h20.joblib` | Random Forest, Level-0 |
| `xgb_level0_h20.joblib` | XGBoost, Level-0 |
| `lstm_level0_h20.keras` | LSTM, Level-0 |
| `meta_rank_bayesridge_h20.joblib` | Linear Bayesian meta-learner, rank target |
| `meta_return_bayesridge_h20.joblib` | Linear Bayesian meta-learner, return target |
| `meta_rank_nonlinear_h20.joblib` | Nonlinear meta-learner, rank target *(selected)* |
| `meta_return_nonlinear_h20.joblib` | Nonlinear meta-learner, return target *(selected)* |
| `meta_quantile_lo_h20.joblib` | 2.5% quantile regressor |
| `meta_quantile_hi_h20.joblib` | 97.5% quantile regressor |
| `pipeline_config_h20.json` | Feature lists, selected meta-learner, holdout metrics |

## How to regenerate them

Open `notebook/` in Kaggle, Colab or Jupyter and run every cell in order. Cell 15 is
the one that writes this folder's contents. A full run takes roughly ten minutes on a
standard CPU instance; no GPU is required.

Then copy the ten files above into this directory and `python predict.py --top 10`
works.

## A caution on reproducing exactly

The notebook downloads live data, so a run today sees a longer history than the run
that produced the published results. The architecture, features and validation scheme
reproduce exactly; the reported IC and backtest figures will shift slightly with the
data window. `pipeline_config_h20.json` records the metrics of the published run so a
re-run can be compared against it.
