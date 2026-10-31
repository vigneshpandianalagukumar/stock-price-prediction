"""Load the trained artefacts and score the current cross-section.

Mirrors Cell 17 of the notebook, which is the only cell that produces a
genuinely live prediction. Cell 16 shows picks from the most recent *holdout*
date -- old enough to already have a realised outcome. This path scores the
latest trading day, which by construction has no outcome yet and cannot be
checked for accuracy until the horizon has elapsed.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

from .config import HORIZON, LOOKBACK, START_DATE, TAG
from .features import (
    ALL_FEATURES,
    LSTM_FEATURES,
    TREE_FEATURES,
    XGB_FEATURES,
    add_rank_features,
    build_panel,
    build_sequences,
    download_macro,
    download_universe,
    live_snapshot,
    meta_matrix,
    neutralise_corporate_actions,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS_DIR = os.path.join(ROOT, "models")
CACHE_DIR = os.path.join(ROOT, "data")


# ---------------------------------------------------------------------------
# Artefact loading
# ---------------------------------------------------------------------------

def load_artifacts(models_dir=MODELS_DIR, tag=TAG):
    """Load every persisted model plus the config written alongside them.

    Verifies that the feature lists recorded at training time match the lists
    this package will build. A mismatch here is the failure mode worth
    guarding: the tree models were fitted on bare NumPy arrays, so scikit-learn
    cannot detect reordered columns at predict time and will return confident
    nonsense instead of raising.
    """
    import joblib

    cfg_path = os.path.join(models_dir, f"pipeline_config_{tag}.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"{cfg_path} not found.\n"
            f"Expected the notebook's Cell 15 artefacts in {models_dir}/ -- "
            f"pipeline_config_{tag}.json, rf_level0_{tag}.joblib, "
            f"xgb_level0_{tag}.joblib, lstm_level0_{tag}.keras and the four "
            "meta-learner files."
        )
    with open(cfg_path) as f:
        cfg = json.load(f)

    for name, expected in [("rf_features", TREE_FEATURES),
                           ("xgb_features", XGB_FEATURES),
                           ("lstm_features", LSTM_FEATURES)]:
        saved = cfg.get(name)
        if saved is not None and list(saved) != list(expected):
            raise RuntimeError(
                f"{name} in {os.path.basename(cfg_path)} does not match "
                f"src/features.py.\n  saved:    {saved}\n  expected: {expected}\n"
                "The models were trained on a different feature layout. Scoring "
                "would silently produce wrong numbers, so this is fatal."
            )

    if cfg.get("horizon") not in (None, HORIZON):
        raise RuntimeError(
            f"Artefacts were trained at horizon {cfg['horizon']} but config.py "
            f"says {HORIZON}."
        )

    def _load(fn):
        p = os.path.join(models_dir, fn)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing artefact: {p}")
        with warnings.catch_warnings():
            # scikit-learn warns loudly if the loading version differs from the
            # fitting version. requirements.txt pins 1.6.1 to avoid it; if a
            # user ignores the pin they should still see the warning once.
            warnings.simplefilter("default")
            return joblib.load(p)

    art = {
        "config": cfg,
        "winner": cfg.get("selected_meta_learner", "nonlinear"),
        "rf": _load(f"rf_level0_{tag}.joblib"),
        "xgb": _load(f"xgb_level0_{tag}.joblib"),
        "meta_rank": _load(f"meta_rank_bayesridge_{tag}.joblib"),
        "meta_ret": _load(f"meta_return_bayesridge_{tag}.joblib"),
        "meta_rank_nl": _load(f"meta_rank_nonlinear_{tag}.joblib"),
        "meta_ret_nl": _load(f"meta_return_nonlinear_{tag}.joblib"),
        "q_lo": _load(f"meta_quantile_lo_{tag}.joblib"),
        "q_hi": _load(f"meta_quantile_hi_{tag}.joblib"),
    }

    from tensorflow import keras
    lstm_path = os.path.join(models_dir, f"lstm_level0_{tag}.keras")
    if not os.path.exists(lstm_path):
        raise FileNotFoundError(f"Missing artefact: {lstm_path}")
    art["lstm"] = keras.models.load_model(lstm_path, compile=False)

    return art


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_live_frame(start=START_DATE, end=None, cache=None, offline=False,
                       verbose=True):
    """Download, engineer, rank and return the live-inference snapshot.

    The whole universe is always processed, even when the caller wants one
    ticker. Features are daily cross-sectional percentile ranks, so a name's
    inputs are defined only relative to the other 230.

    Set `cache` to a CSV path to save the raw download, and `offline=True` to
    read it back without touching the network.
    """
    cache = cache or os.path.join(CACHE_DIR, "raw_prices.csv")

    if offline:
        if not os.path.exists(cache):
            raise FileNotFoundError(
                f"--offline needs a cached snapshot at {cache}. Run once online "
                "first, or copy the file across."
            )
        if verbose:
            print(f"Reading cached prices from {cache}")
        df_px = pd.read_csv(cache, parse_dates=["Date"])
        macro_path = cache.replace(".csv", "_macro.csv")
        macro = pd.read_csv(macro_path, parse_dates=["Date"]).set_index("Date")
    else:
        if verbose:
            print("Downloading equity data ...")
        df_px, failed = download_universe(start=start, end=end, verbose=verbose)
        if verbose:
            print(f"Usable tickers : {df_px['Ticker'].nunique()}")
            if failed:
                print(f"Dropped        : {len(failed)}  {failed}")
        macro = download_macro(start=start, end=end)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df_px.to_csv(cache, index=False)
        macro.reset_index().to_csv(cache.replace(".csv", "_macro.csv"), index=False)

    df_px = neutralise_corporate_actions(df_px, verbose=verbose)
    df = build_panel(df_px, macro, verbose=verbose)
    df = add_rank_features(df)
    df_live = live_snapshot(df, LOOKBACK)

    if verbose:
        print(f"Live snapshot: {df_live['Date'].nunique()} dates, "
              f"{df_live['Ticker'].nunique()} tickers, {len(df_live):,} rows")
    return df_live


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def predict(df_live, art, verbose=True):
    """Score the single most recent date in `df_live`. Returns a DataFrame."""
    X3D_live, POS_live = build_sequences(df_live, LSTM_FEATURES, LOOKBACK)

    _live_dates = df_live.loc[POS_live, "Date"].to_numpy()
    _as_of = _live_dates.max()
    _final = _live_dates == _as_of

    live_idx = POS_live[_final]
    X2D_RF_live = df_live.loc[live_idx, TREE_FEATURES].to_numpy(np.float32)
    X2D_XGB_live = df_live.loc[live_idx, XGB_FEATURES].to_numpy(np.float32)
    X3D_live_f = X3D_live[_final]
    live_tickers = df_live.loc[live_idx, "Ticker"].to_numpy()
    live_vixz = df_live.loc[live_idx, "VIX_Z"].to_numpy(np.float32)

    l_rf = art["rf"].predict(X2D_RF_live)
    l_xgb = art["xgb"].predict(X2D_XGB_live)
    l_lstm = art["lstm"].predict(X3D_live_f, batch_size=4096, verbose=0).ravel()

    M_live = meta_matrix(l_rf, l_xgb, l_lstm, live_vixz)

    if art["winner"] == "nonlinear":
        l_pred_rank = art["meta_rank_nl"].predict(M_live)
        l_pred_ret = art["meta_ret_nl"].predict(M_live)
        l_lo, l_hi = art["q_lo"].predict(M_live), art["q_hi"].predict(M_live)
        l_lo, l_hi = np.minimum(l_lo, l_hi), np.maximum(l_lo, l_hi)
    else:
        l_pred_rank, _ = art["meta_rank"].predict(M_live, return_std=True)
        l_pred_ret, l_std_ret = art["meta_ret"].predict(M_live, return_std=True)
        l_lo = l_pred_ret - 1.96 * l_std_ret
        l_hi = l_pred_ret + 1.96 * l_std_ret

    cfg = art["config"]
    out = pd.DataFrame({
        "Ticker": live_tickers,
        "Horizon": HORIZON,
        "HorizonLabel": (f"~{HORIZON / 252:.2g} yr" if HORIZON >= 100
                         else f"~{HORIZON} trading days"),
        "PredRank": l_pred_rank,
        "PredRet": l_pred_ret,
        "VaR95_Lo": l_lo,
        "VaR95_Hi": l_hi,
        "NCycles": cfg.get("n_independent_cycles", np.nan),
        "LowSampleWarning": bool(cfg.get("low_sample_warning", False)),
        "AsOfDate": pd.Timestamp(_as_of).date(),
    }).sort_values("PredRank", ascending=False).reset_index(drop=True)

    if verbose:
        print(f"Scored {len(out)} tickers as of {pd.Timestamp(_as_of).date()} "
              f"(meta-learner: {art['winner']})")
    return out


def run(start=START_DATE, end=None, offline=False, cache=None,
        models_dir=MODELS_DIR, verbose=True):
    """Full path: artefacts -> data -> features -> predictions."""
    art = load_artifacts(models_dir=models_dir)
    df_live = prepare_live_frame(start=start, end=end, cache=cache,
                                 offline=offline, verbose=verbose)
    return predict(df_live, art, verbose=verbose)


def lookup(results, query):
    """Filter predictions by ticker fragment, ignoring the .NS suffix."""
    q = query.strip().upper().replace(" ", "")
    return results[
        results["Ticker"].str.upper().str.replace(".NS", "", regex=False)
        .str.contains(q, na=False)
    ]
