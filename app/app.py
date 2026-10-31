"""
Streamlit results console for the Bayesian stacking equity forecaster.

Loads persisted prediction outputs and figures and serves them for review.
No training or inference happens here — the notebook produces the artefacts.

Run from the repository root:
    streamlit run app/app.py
"""

import glob
import os

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------
# Works whether the repo uses subfolders (results/, figures/) or everything
# sits flat in one directory, so the app runs unchanged in both layouts.

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEARCH_DIRS = [
    os.path.join(ROOT, "results"),
    os.path.join(ROOT, "figures"),
    ROOT,
    HERE,
    os.getcwd(),
]


def find(filename):
    """Return the first existing path for `filename` across the search dirs."""
    for d in SEARCH_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None


def find_glob(pattern):
    """Return all matches for `pattern` across the search dirs."""
    hits = []
    for d in SEARCH_DIRS:
        hits.extend(glob.glob(os.path.join(d, pattern)))
    seen, out = set(), []
    for h in hits:
        base = os.path.basename(h)
        if base not in seen:
            seen.add(base)
            out.append(h)
    return sorted(out)


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Bayesian Stacking — Equity Forecast",
    layout="wide",
    page_icon="chart_with_upwards_trend",
)

st.title("Regime-Adaptive Bayesian Stacking for Equity Forecasting")
st.caption(
    "Random Forest, XGBoost and LSTM combined under a Bayesian meta-learner with "
    "regime-conditioned weights and calibrated risk bounds. Validated under purged "
    "walk-forward cross-validation."
)

summary_path = find("summary_h20.csv")

# Headline metrics are read from the run's own summary file rather than
# hardcoded, so the page can never disagree with the artefacts beside it.
_m = {}
if summary_path:
    _s = pd.read_csv(summary_path, index_col=0)
    _col = _s.columns[0]
    _m = {k: _s.loc[k, _col] for k in _s.index}
    with st.expander("Full summary metrics"):
        st.dataframe(_s.reset_index(), use_container_width=True)


def _num(key, fmt, fallback="—"):
    try:
        return format(float(_m[key]), fmt)
    except (KeyError, TypeError, ValueError):
        return fallback


c1, c2, c3 = st.columns(3)
c1.metric("Mean cross-sectional IC", _num("Mean_IC", ".4f"),
          help="Typical literature range: 0.02–0.05")
c2.metric("95% interval coverage (PICP)", _num("PICP_Pct", ".1f") + "%",
          help="Target: 95%")
c3.metric("Long–short Sharpe", _num("LongShort_Sharpe", ".3f"),
          help="Net of 10 bps per side")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Ask about a stock", "Diagnostics", "Backtest equity curve", "Interpretability (SHAP)"]
)

# --------------------------------------------------------------------------
# Tab 1 — live prediction lookup
# --------------------------------------------------------------------------
with tab1:
    st.subheader("Query the model")

    # ---- live scoring ----------------------------------------------------
    # Loads the saved artefacts and scores the current cross-section from
    # scratch. Everything below this block reads the stored CSV instead.
    with st.expander("Score today's market now (downloads fresh prices)"):
        st.caption(
            "Downloads the full 231-name universe and re-runs the pipeline. "
            "Takes several minutes. Needed because the features are daily "
            "cross-sectional ranks — a stock's inputs are defined only "
            "relative to the rest of the universe, so one name cannot be "
            "scored on its own."
        )
        if st.button("Run live inference"):
            try:
                import sys

                sys.path.insert(0, ROOT)
                from src.inference import run as run_inference

                with st.spinner("Downloading and scoring…"):
                    fresh = run_inference(verbose=False)
                fresh.to_csv(os.path.join(ROOT, "results", "live_predictions_h20.csv"),
                             index=False)
                st.success(f"Scored {len(fresh)} tickers as of {fresh['AsOfDate'].iloc[0]}.")
                st.dataframe(fresh, use_container_width=True, height=320)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Inference failed: {exc}")

    files = find_glob("live_predictions_h*.csv")
    if not files:
        st.warning(
            "No live_predictions_h*.csv found. Expected in results/ or the repo root."
        )
    else:
        live = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

        as_of = live["AsOfDate"].iloc[0]
        st.caption(
            f"Predictions as of **{as_of}** — live inference on the most recent "
            "trading date with complete features. The forward window has not yet "
            "elapsed, so these cannot be scored for accuracy."
        )

        horizons = sorted(live["Horizon"].unique().tolist())
        if len(horizons) > 1:
            pick = st.radio(
                "Horizon",
                horizons,
                horizontal=True,
                format_func=lambda h: live[live.Horizon == h]["HorizonLabel"].iloc[0],
            )
            live = live[live["Horizon"] == pick]

        n_pos = int((live["PredRet"] > 0).sum())
        n_neg = int((live["PredRet"] < 0).sum())
        s1, s2, s3 = st.columns(3)
        s1.metric("Predicted to outperform", f"{n_pos} stocks")
        s2.metric("Predicted to underperform", f"{n_neg} stocks")
        s3.metric(
            "Return range",
            f"{live['PredRet'].min() * 100:.2f}% to {live['PredRet'].max() * 100:+.2f}%",
        )

        query = st.text_input("Enter an NSE ticker (e.g. SBIN, RELIANCE, INFY)", "")

        if query:
            q = query.strip().upper().replace(" ", "")
            hits = live[
                live["Ticker"]
                .str.upper()
                .str.replace(".NS", "", regex=False)
                .str.contains(q, na=False)
            ]
            if len(hits) == 0:
                st.error(
                    f"No match for '{query}' among the "
                    f"{live['Ticker'].nunique()} stocks in the trained universe."
                )
            for _, row in hits.iterrows():
                with st.container(border=True):
                    st.markdown(f"### {row['Ticker']}")
                    st.caption(f"As of {row['AsOfDate']} · {row['HorizonLabel']} horizon")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted relative return", f"{row['PredRet'] * 100:+.2f}%")
                    m2.metric(
                        "95% interval",
                        f"[{row['VaR95_Lo'] * 100:+.1f}%, {row['VaR95_Hi'] * 100:+.1f}%]",
                    )
                    m3.metric("Cross-sectional rank", f"{row['PredRank']:.4f}")
                    st.caption(
                        "Return is relative to the stock's sector peers, not absolute. "
                        "Higher rank means greater expected outperformance."
                    )
                    if row.get("LowSampleWarning", False):
                        st.warning(
                            f"Only ~{row['NCycles']:.0f} independent "
                            f"{row['Horizon']}-day cycles support this horizon — "
                            "treat as directional, not precise."
                        )
        else:
            view = st.radio(
                "Show",
                ["Top 20 (outperformers)", "Bottom 20 (underperformers)", "Full ranking"],
                horizontal=True,
            )
            if view.startswith("Top"):
                shown = live.nlargest(20, "PredRank")
            elif view.startswith("Bottom"):
                shown = live.nsmallest(20, "PredRank")
            else:
                shown = live.sort_values("PredRank", ascending=False)
            st.dataframe(shown, use_container_width=True, height=420)

# --------------------------------------------------------------------------
# Tab 2 — diagnostics
# --------------------------------------------------------------------------
with tab2:
    st.subheader("Model diagnostics")
    diag = find("diagnostic_plots_h20.png")
    if diag:
        st.image(diag, use_container_width=True)
        st.caption(
            "Clockwise from top left: holdout equity curves; daily cross-sectional "
            "Information Coefficient; interval calibration; effective meta-weight "
            "versus volatility regime."
        )
    else:
        st.info("diagnostic_plots_h20.png not found.")

    picks = find("todays_picks_h20.png")
    if picks:
        st.subheader("Latest ranked picks")
        st.image(picks, use_container_width=True)

# --------------------------------------------------------------------------
# Tab 3 — backtest
# --------------------------------------------------------------------------
with tab3:
    st.subheader("Holdout backtest — growth of 1")
    bt_path = find("backtest_h20.csv")
    if bt_path:
        bt = pd.read_csv(bt_path)
        date_col = bt.columns[0]
        bt[date_col] = pd.to_datetime(bt[date_col])
        cols = [
            c
            for c in ["LongShort", "LongOnly", "Benchmark"]
            if c in bt.columns and pd.api.types.is_numeric_dtype(bt[c])
        ]
        if cols:
            st.line_chart((1 + bt.set_index(date_col)[cols]).cumprod())
            st.caption("Net of 10 basis points transaction cost per side.")
    else:
        st.info("backtest_h20.csv not found.")

    hold = find("holdout_predictions_h20.csv")
    if hold:
        st.subheader("Holdout predictions")
        st.dataframe(pd.read_csv(hold), use_container_width=True, height=350)

# --------------------------------------------------------------------------
# Tab 4 — SHAP
# --------------------------------------------------------------------------
with tab4:
    st.subheader("Feature attribution")
    col_a, col_b = st.columns(2)
    sm = find("shap_summary_h20.png")
    sb = find("shap_bar_h20.png")
    if sm:
        col_a.image(sm, use_container_width=True)
    if sb:
        col_b.image(sb, use_container_width=True)
    if not (sm or sb):
        st.info("SHAP figures not found.")

st.divider()
st.caption(
    "Two-tier Bayesian hierarchical stacking · Level-0: Random Forest, XGBoost, LSTM · "
    "Level-1: Bayesian meta-learner with regime-conditioned weights · "
    "Academic project — not investment advice."
)
