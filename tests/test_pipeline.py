"""Regression tests.

The first group runs anywhere with no network and no model files. It guards
feature ordering, which is the failure this repository is most exposed to: the
tree models were fitted on NumPy arrays, so a reordered column produces wrong
numbers rather than an exception.

The second group is skipped unless the artefacts and a reference prediction
file are present. When they are, it re-scores the cached snapshot and checks
the output still matches the numbers the notebook produced.

    pip install pytest
    pytest -q
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src import features as F
from src.config import LOOKBACK, TAG
from src.universe import SECTOR_MAP, TICKER_SECTOR, TICKERS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")
REFERENCE = os.path.join(RESULTS, f"live_predictions_{TAG}.csv")
CACHE = os.path.join(ROOT, "data", "raw_prices.csv")


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def test_universe_size():
    assert len(TICKERS) == 231
    assert len(SECTOR_MAP) == 17
    assert len(TICKER_SECTOR) == len(TICKERS)


def test_every_ticker_has_ns_suffix():
    assert all(t.endswith(".NS") for t in TICKERS)


def test_no_duplicate_tickers():
    assert len(set(TICKERS)) == len(TICKERS)


# ---------------------------------------------------------------------------
# Feature ordering -- the important ones
# ---------------------------------------------------------------------------

def test_tree_feature_order_exact():
    assert F.TREE_FEATURES == [
        "RSI14_R", "MACD_N_R", "DIST_SMA200_R", "DIST_SMA50_R", "MOM_12_1_R",
        "REV_5_R", "ATR_N_R", "BB_W_R", "VOL_60_R", "OBV_N_R", "TURN_R_R",
        "VIX_Z", "VIX_CHG5", "MKT_TREND",
    ]


def test_xgb_features_extend_tree_features():
    assert len(F.XGB_FEATURES) == 25
    assert F.XGB_FEATURES[:14] == F.TREE_FEATURES
    assert all(c.endswith("_CHG5") for c in F.XGB_FEATURES[14:])


def test_lstm_features_exclude_volume():
    assert len(F.LSTM_FEATURES) == 12
    assert "OBV_N_R" not in F.LSTM_FEATURES
    assert "TURN_R_R" not in F.LSTM_FEATURES


def test_feature_counts_match_model_input_shapes():
    """The counts the saved models expect: 14, 25, 12."""
    assert (len(F.TREE_FEATURES), len(F.XGB_FEATURES), len(F.LSTM_FEATURES)) == (14, 25, 12)


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------

def _toy_frame(n_tickers=3, n_days=40):
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rng = np.random.RandomState(0)
    for i in range(n_tickers):
        for d in dates:
            rows.append({"Ticker": f"T{i}.NS", "Date": d,
                         **{c: rng.rand() for c in F.LSTM_FEATURES}})
    return pd.DataFrame(rows).sort_values(["Ticker", "Date"]).reset_index(drop=True)


def test_build_sequences_shape_and_alignment():
    frame = _toy_frame()
    X, pos = F.build_sequences(frame, F.LSTM_FEATURES, LOOKBACK)
    assert X.shape[1:] == (LOOKBACK, len(F.LSTM_FEATURES))
    assert X.shape[0] == len(pos)
    # Each window must end on its own row.
    last_row = frame.loc[pos[0], F.LSTM_FEATURES].to_numpy(np.float32)
    assert np.allclose(X[0, -1, :], last_row)


def test_build_sequences_does_not_cross_tickers():
    frame = _toy_frame(n_tickers=2, n_days=LOOKBACK + 1)
    X, pos = F.build_sequences(frame, F.LSTM_FEATURES, LOOKBACK)
    # Two sequences per ticker, four in total -- never a window spanning both.
    assert len(pos) == 4
    assert frame.loc[pos, "Ticker"].value_counts().tolist() == [2, 2]


def test_meta_matrix_layout():
    rf = np.array([0.1, 0.2])
    xgb = np.array([0.3, 0.4])
    lstm = np.array([0.5, 0.6])
    vix = np.array([1.0, -2.0])
    M = F.meta_matrix(rf, xgb, lstm, vix)
    assert M.shape == (2, 7)
    assert np.allclose(M[:, :3], np.column_stack([rf, xgb, lstm]))
    assert np.allclose(M[:, 3], vix)
    assert np.allclose(M[:, 4], rf * vix)
    assert np.allclose(M[:, 6], lstm * vix)
    assert len(F.META_NAMES) == 7


def test_rank_features_are_centred_on_zero():
    """Percentile rank minus 0.5 must span (-0.5, +0.5) within each date."""
    rng = np.random.RandomState(1)
    df = pd.DataFrame({
        "Date": np.repeat(pd.bdate_range("2024-01-01", periods=3), 50),
        "Ticker": np.tile([f"T{i}.NS" for i in range(50)], 3),
        **{c: rng.rand(150) for c in F.RAW_FEATURES},
    })
    out = F.add_rank_features(df.copy())
    per_day = out.groupby("Date")["RSI14_R"]
    assert per_day.min().max() < 0
    assert per_day.max().min() > 0
    # rank(pct=True) runs 1/n .. n/n, so the centred mean sits at 1/(2n)
    # rather than exactly zero. Harmless, but worth pinning so a future
    # change to the normalisation is noticed.
    assert abs(per_day.mean().mean() - 1 / (2 * 50)) < 1e-9


def test_corporate_action_guard_removes_structural_break():
    dates = pd.bdate_range("2024-01-01", periods=10)
    close = [100.0] * 5 + [20.0] + [100.0] * 4      # a 1:5 split, unadjusted
    df = pd.DataFrame({"Ticker": "X.NS", "Date": dates, "Close": close,
                       "High": close, "Low": close, "Volume": 1000})
    out = F.neutralise_corporate_actions(df, verbose=False)
    assert out.loc[5, "Close"] == 100.0             # forward-filled, not kept
    assert out["Close"].pct_change().abs().max(skipna=True) < 0.35


# ---------------------------------------------------------------------------
# End-to-end -- only when artefacts and a reference file exist
# ---------------------------------------------------------------------------

_can_run_e2e = (
    os.path.exists(os.path.join(MODELS, f"pipeline_config_{TAG}.json"))
    and os.path.exists(REFERENCE)
    and os.path.exists(CACHE)
)

e2e = pytest.mark.skipif(
    not _can_run_e2e,
    reason="needs models/, results/live_predictions_*.csv and data/raw_prices.csv",
)


@e2e
def test_artifact_config_matches_feature_lists():
    from src.inference import load_artifacts
    art = load_artifacts()                       # raises on any mismatch
    assert art["winner"] in {"linear", "nonlinear"}


@e2e
def test_reproduces_notebook_predictions():
    """Re-score the cached snapshot and compare against the notebook's output.

    Tolerance is loose on purpose: TensorFlow is not bit-reproducible across
    machines. Anything beyond this is a real break, not float noise.
    """
    from src.inference import load_artifacts, predict, prepare_live_frame

    ref = pd.read_csv(REFERENCE).set_index("Ticker").sort_index()
    art = load_artifacts()
    df_live = prepare_live_frame(offline=True, verbose=False)
    got = predict(df_live, art, verbose=False).set_index("Ticker").sort_index()

    common = ref.index.intersection(got.index)
    assert len(common) > 200, f"only {len(common)} tickers in common"

    for col in ["PredRank", "PredRet", "VaR95_Lo", "VaR95_Hi"]:
        np.testing.assert_allclose(
            got.loc[common, col].to_numpy(),
            ref.loc[common, col].to_numpy(),
            atol=2e-3,
            err_msg=f"{col} drifted from the notebook's output",
        )
