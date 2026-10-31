"""Feature construction for inference.

Every function here is lifted from Cells 3-6 of the training notebook without
alteration to the arithmetic. This file is deliberately not "cleaned up".

The reason is specific. The Random Forest and XGBoost artefacts were fitted on
NumPy arrays rather than DataFrames, so they carry no `feature_names_in_`
attribute. scikit-learn therefore cannot check at predict time that the columns
arriving are the columns it was trained on. Hand it fourteen floats in the
wrong order and it returns fourteen plausible-looking numbers that are wrong.
There is no exception and no warning.

Column order is therefore part of the model. TREE_FEATURES, XGB_FEATURES and
LSTM_FEATURES below must stay exactly as written.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from .config import (
    CORP_ACTION_THRESH,
    HORIZON,
    MACRO_INDEX,
    MACRO_VIX,
    MIN_HISTORY,
    MIN_NAMES_PER_DAY,
    START_DATE,
)
from .universe import TICKER_SECTOR, TICKERS

# ---------------------------------------------------------------------------
# Feature name lists -- Cell 5. Order is load-bearing. Do not sort these.
# ---------------------------------------------------------------------------

RAW_FEATURES = [
    "RSI14", "MACD_N", "DIST_SMA200", "DIST_SMA50", "MOM_12_1",
    "REV_5", "ATR_N", "BB_W", "VOL_60", "OBV_N", "TURN_R",
]

RANK_FEATURES = [c + "_R" for c in RAW_FEATURES]
RANK_CHG_FEATURES = [rc + "_CHG5" for rc in RANK_FEATURES]
MACRO_FEATURES = ["VIX_Z", "VIX_CHG5", "MKT_TREND"]

# Random Forest: 14 columns
TREE_FEATURES = RANK_FEATURES + MACRO_FEATURES

# XGBoost: the same 14, then the 11 five-day rank deltas -> 25 columns
XGB_FEATURES = TREE_FEATURES + RANK_CHG_FEATURES

# LSTM: 12 columns, volume features excluded
LSTM_FEATURES = [
    "RSI14_R", "MACD_N_R", "DIST_SMA200_R", "DIST_SMA50_R", "MOM_12_1_R",
    "REV_5_R", "ATR_N_R", "BB_W_R", "VOL_60_R",
] + MACRO_FEATURES

ALL_FEATURES = sorted(set(TREE_FEATURES) | set(XGB_FEATURES) | set(LSTM_FEATURES))


# ---------------------------------------------------------------------------
# Download -- Cell 3
# ---------------------------------------------------------------------------

def _download_batch(tickers, start, end, min_history=MIN_HISTORY):
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      group_by="ticker", progress=False, threads=True)
    out, bad = {}, []
    for t in tickers:
        try:
            sub = raw[t].dropna(how="all")
        except (KeyError, TypeError):
            bad.append(t)
            continue
        if len(sub) < min_history:
            bad.append(t)
            continue
        out[t] = sub
    return out, bad


def download_universe(tickers=None, start=START_DATE, end=None,
                      batch_size=40, retry_pause=2.0, min_history=MIN_HISTORY,
                      verbose=True):
    """Daily OHLCV for the universe, split-adjusted by yfinance.

    `auto_adjust=True` is mandatory. Without it a 1:5 split enters the data as
    a -80% day and poisons every feature computed downstream.
    """
    import time

    tickers = list(TICKERS) if tickers is None else list(tickers)
    if end is None:
        end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    frames, failed = [], []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        if verbose:
            print(f"  batch {i // batch_size + 1}: {len(batch)} tickers ...", end=" ")

        good, bad = _download_batch(batch, start, end, min_history)

        if bad:
            time.sleep(retry_pause)
            still_bad = []
            for t in bad:
                g2, _ = _download_batch([t], start, end, min_history)
                if t in g2:
                    good[t] = g2[t]
                else:
                    still_bad.append(t)
                time.sleep(0.4)
            bad = still_bad

        for t, sub in good.items():
            d = sub.reset_index()[["Date", "High", "Low", "Close", "Volume"]].copy()
            d["Ticker"] = t
            frames.append(d)

        failed.extend(bad)
        if verbose:
            print(f"{len(good)} ok" + (f", {len(bad)} failed after retry" if bad else ""))

    if not frames:
        raise RuntimeError(
            "yfinance returned no usable data for any ticker. Yahoo rate-limits "
            "aggressively; wait a minute and retry, or run with --offline against "
            "a cached snapshot."
        )
    return pd.concat(frames, ignore_index=True), failed


def download_macro(start=START_DATE, end=None):
    """India VIX and Nifty 50, reduced to the three macro features.

    The z-score is EXPANDING, not full-sample, so no future information enters
    it -- and so its value on a given date depends on `start`. Keep `start` at
    the training value or the feature stops matching what the models saw.
    """
    import yfinance as yf

    if end is None:
        end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    def fetch_series(ticker, name):
        r = yf.download(ticker, start=start, end=end,
                        auto_adjust=True, progress=False)["Close"]
        s = r.iloc[:, 0] if isinstance(r, pd.DataFrame) else r
        s.name = name
        return s

    vix = fetch_series(MACRO_VIX, "VIX")
    nifty = fetch_series(MACRO_INDEX, "NIFTY")
    macro = pd.concat([vix, nifty], axis=1).ffill().dropna()

    macro["VIX_Z"] = ((macro["VIX"] - macro["VIX"].expanding(252).mean())
                      / macro["VIX"].expanding(252).std())
    macro["VIX_CHG5"] = macro["VIX"].pct_change(5)
    macro["MKT_TREND"] = macro["NIFTY"] / macro["NIFTY"].rolling(200).mean() - 1

    macro = macro[["VIX_Z", "VIX_CHG5", "MKT_TREND"]].dropna()
    macro.index.name = "Date"
    return macro


# ---------------------------------------------------------------------------
# Corporate-action guard + feature engineering -- Cell 4
# ---------------------------------------------------------------------------

def neutralise_corporate_actions(df_px, verbose=True):
    """Blank any daily move beyond 35% and forward-fill through it.

    NSE circuit limits make genuine single-day moves above 35% rare, so a move
    that large is almost always a split, bonus or demerger that yfinance's
    adjustment did not catch -- structural, not informational.
    """
    df_px = df_px.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    _ret = df_px.groupby("Ticker")["Close"].pct_change()
    _break = _ret.abs() > CORP_ACTION_THRESH

    if verbose:
        print(f"Corporate-action breaks neutralised: {_break.sum()} rows across "
              f"{df_px.loc[_break, 'Ticker'].nunique()} symbols")

    df_px.loc[_break, "Close"] = np.nan
    df_px["Close"] = df_px.groupby("Ticker")["Close"].ffill()
    return df_px[df_px["Close"].notna()].reset_index(drop=True)


def engineer(g):
    """Eleven stationary technical features, per ticker. Verbatim from Cell 4."""
    g = g.sort_values("Date").copy()
    c, h, l, v = g["Close"], g["High"], g["Low"], g["Volume"]

    # Momentum / trend
    g["RSI14"] = ta.momentum.rsi(c, window=14)
    g["MACD_N"] = ta.trend.macd_diff(c) / c
    g["DIST_SMA200"] = c / c.rolling(200).mean() - 1
    g["DIST_SMA50"] = c / c.rolling(50).mean() - 1
    g["MOM_12_1"] = c.shift(21) / c.shift(252) - 1
    g["REV_5"] = c / c.shift(5) - 1

    # Volatility
    g["ATR_N"] = ta.volatility.average_true_range(h, l, c, window=14) / c
    bb_h = ta.volatility.bollinger_hband(c)
    bb_l = ta.volatility.bollinger_lband(c)
    bb_m = ta.volatility.bollinger_mavg(c)
    g["BB_W"] = (bb_h - bb_l) / bb_m
    g["VOL_60"] = c.pct_change().rolling(60).std()

    # Volume / flow
    obv = ta.volume.on_balance_volume(c, v)
    g["OBV_N"] = (obv - obv.shift(20)) / (v.rolling(20).sum() + 1.0)
    g["TURN_R"] = v.rolling(20).mean() / (v.rolling(120).mean() + 1.0)

    # Forward return. NaN for the most recent HORIZON rows by construction,
    # which is exactly the region live inference operates in.
    g["FWD_RET"] = c.shift(-HORIZON) / c - 1
    return g


def build_panel(df_px, macro, verbose=True):
    """Engineered panel joined to macro, with sector labels attached."""
    df = pd.concat([engineer(g) for _, g in df_px.groupby("Ticker", sort=False)],
                   ignore_index=True)
    df = df.merge(macro, left_on="Date", right_index=True, how="inner")
    df["SECTOR"] = df["Ticker"].map(TICKER_SECTOR)

    # FWD_RET is allowed to stay NaN -- recent rows have features but no
    # realised outcome yet, and those are the rows we predict on.
    df = df.dropna(subset=RAW_FEATURES + ["VIX_Z"]).reset_index(drop=True)

    counts = df.groupby("Date")["Ticker"].transform("size")
    df = df[counts >= MIN_NAMES_PER_DAY].reset_index(drop=True)

    if verbose:
        print(f"Rows after feature engineering: {len(df):,}")
        print(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    return df


# ---------------------------------------------------------------------------
# Cross-sectional rank normalisation -- Cell 5
# ---------------------------------------------------------------------------

def add_rank_features(df):
    """Convert each raw feature to a daily cross-sectional percentile rank.

    This is why single-ticker inference is impossible. RSI14_R is not a
    property of the stock; it is that stock's position among the other 230 on
    that date. Score one name alone and every rank collapses to 0.0.
    """
    for c in RAW_FEATURES:
        df[c + "_R"] = df.groupby("Date")[c].rank(pct=True) - 0.5
    for rc in RANK_FEATURES:
        df[rc + "_CHG5"] = df.groupby("Ticker")[rc].diff(5)
    return df


def live_snapshot(df, lookback):
    """The most recent dates on which every feature is present.

    Taken before any TARGET filter, because the rows worth predicting are
    precisely the ones whose forward window has not elapsed.
    """
    _live_ok = df[ALL_FEATURES].notna().all(axis=1)
    _recent_dates = np.array(sorted(df.loc[_live_ok, "Date"].unique()))[-(lookback + 10):]
    out = df[_live_ok & df["Date"].isin(_recent_dates)].copy()
    return out.sort_values(["Ticker", "Date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sequence tensor -- Cell 6
# ---------------------------------------------------------------------------

def build_sequences(frame, feat_cols, lookback):
    """Rolling LSTM windows, built independently per ticker.

    Requires `frame` to carry a clean 0..n-1 RangeIndex, because `feat_all` is
    indexed positionally by the group's index values. `live_snapshot` resets
    the index for this reason -- do not pass a filtered frame straight in.

    Returns (X3D, row_pos) where row_pos gives the dataframe row each sequence
    ENDS on, which is the row being predicted.
    """
    windows, row_pos = [], []
    feat_all = frame[feat_cols].to_numpy(np.float32)

    for _, g in frame.groupby("Ticker", sort=False):
        idx = g.index.to_numpy()
        if len(idx) < lookback:
            continue
        block = feat_all[idx]
        w = np.lib.stride_tricks.sliding_window_view(
            block, (lookback, block.shape[1]))[:, 0]
        windows.append(np.ascontiguousarray(w))
        row_pos.append(idx[lookback - 1:])

    if not windows:
        raise RuntimeError(
            f"No ticker had the {lookback} consecutive rows needed to form a "
            "sequence. The download window is probably too short."
        )

    return (np.concatenate(windows, axis=0).astype(np.float32),
            np.concatenate(row_pos, axis=0))


# ---------------------------------------------------------------------------
# Meta-matrix -- Cell 10
# ---------------------------------------------------------------------------

META_NAMES = ["RF", "XGB", "LSTM", "VIX_Z", "RF x VIX", "XGB x VIX", "LSTM x VIX"]


def meta_matrix(p_rf, p_xgb, p_lstm, vix_z):
    """Seven meta-features: three base predictions, the regime, their products.

    The interaction block is what lets the meta-learner reweight base models by
    volatility regime rather than applying one fixed blend everywhere.
    """
    base = np.column_stack([p_rf, p_xgb, p_lstm])
    inter = base * vix_z.reshape(-1, 1)
    return np.column_stack([base, vix_z, inter])
