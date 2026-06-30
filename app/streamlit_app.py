"""Streamlit dashboard for the weather forecast MLOps pipeline.

Shows model health across all three cities: current model version,
training history, and rolling prediction accuracy over time.

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MONITORING_DIR = PROJECT_ROOT / "monitoring"
REGISTRY_PATH = MODELS_DIR / "registry.json"

CITIES = ["mountain_view", "san_francisco", "los_angeles"]
CITY_LABELS = {
    "mountain_view": "Mountain View",
    "san_francisco": "San Francisco",
    "los_angeles": "Los Angeles",
}

DRIFT_THRESHOLD_MULTIPLIER = 1.5


@st.cache_data(ttl=300)
def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


@st.cache_data(ttl=300)
def load_monitoring_log(city: str) -> pd.DataFrame:
    path = MONITORING_DIR / f"{city}_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["ds"])
    return df.sort_values("ds")


def add_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1rem;
            max-width: 95%;
        }
        h1 { font-size: 2.6rem !important; }
        div[data-testid="stMetricValue"] {
            font-size: 2.4rem;
        }
        .status-ok {
            color: #21c55d;
            font-weight: 700;
        }
        .status-warn {
            color: #f59e0b;
            font-weight: 700;
        }
        .status-card {
            border: 1px solid rgba(250,250,250,0.15);
            border-radius: 14px;
            padding: 18px;
            background: rgba(255,255,255,0.035);
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_latest_entry(registry: dict, city: str) -> dict | None:
    entries = [m for m in registry["models"] if m["city"] == city]
    return entries[-1] if entries else None


def draw_error_trend(df: pd.DataFrame, city_label: str, baseline_mae: float):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["ds"], df["abs_error"], marker="o", markersize=4, color="#3b82f6", label="Daily absolute error")

    # Rolling 7-day average for a smoother trend line
    if len(df) >= 3:
        rolling = df["abs_error"].rolling(window=min(7, len(df)), min_periods=1).mean()
        ax.plot(df["ds"], rolling, color="#ef4444", linewidth=2, label="7-day rolling avg")

    ax.axhline(baseline_mae, color="#94a3b8", linestyle="--", linewidth=1, label=f"Training MAE ({baseline_mae:.2f}°C)")
    ax.axhline(baseline_mae * DRIFT_THRESHOLD_MULTIPLIER, color="#f59e0b", linestyle=":", linewidth=1,
               label=f"Drift threshold ({baseline_mae * DRIFT_THRESHOLD_MULTIPLIER:.2f}°C)")

    ax.set_ylabel("Absolute error (°C)", fontsize=11)
    ax.set_title(f"{city_label} — prediction error over time", fontsize=13, weight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="Weather Forecast MLOps Dashboard",
        page_icon="🌤️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    add_css()

    st.title("🌤️ Weather Forecast MLOps Dashboard")
    st.caption(
        "Daily max temperature forecasting for three California cities. "
        "Models retrain automatically every day via GitHub Actions, with "
        "rolling drift monitoring against fresh observed data."
    )

    registry = load_registry()

    with st.sidebar:
        st.header("City")
        city = st.selectbox("Select city", CITIES, format_func=lambda c: CITY_LABELS[c])

    latest = get_latest_entry(registry, city)
    if latest is None:
        st.warning(f"No trained model found for {CITY_LABELS[city]}.")
        return

    log = load_monitoring_log(city)

    top_left, top_right = st.columns([1, 1])

    with top_left:
        st.subheader("Model status")

        if len(log) > 0:
            rolling_mae = log["abs_error"].tail(7).mean()
            drift = rolling_mae > (latest["validation_mae_c"] * DRIFT_THRESHOLD_MULTIPLIER)
            status_class = "status-warn" if drift else "status-ok"
            status_text = "⚠ Drift detected" if drift else "✓ Healthy"
        else:
            rolling_mae = None
            status_class = "status-ok"
            status_text = "No monitoring data yet"

        st.markdown(
            f"""
            <div class="status-card">
                <div><b>Status:</b> <span class="{status_class}">{status_text}</span></div>
                <div><b>Current model version:</b> {latest['version']}</div>
                <div><b>Trained at:</b> {latest['trained_at']}</div>
                <div><b>Training data:</b> {latest['data_start']} to {latest['data_end']} ({latest['train_rows']:,} rows)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Training MAE", f"{latest['validation_mae_c']:.2f}°C")
        c2.metric(
            "Recent 7-day MAE",
            f"{rolling_mae:.2f}°C" if rolling_mae is not None else "—",
        )
        n_versions = len([m for m in registry["models"] if m["city"] == city])
        c3.metric("Model versions", n_versions)

        st.subheader("Training history")
        history = [m for m in registry["models"] if m["city"] == city]
        history_df = pd.DataFrame(history)[["version", "trained_at", "validation_mae_c", "train_rows"]]
        st.dataframe(history_df.iloc[::-1], width="stretch", hide_index=True)

    with top_right:
        if len(log) > 0:
            fig = draw_error_trend(log, CITY_LABELS[city], latest["validation_mae_c"])
            st.pyplot(fig, width="stretch")
        else:
            st.info("No monitoring history yet. The drift evaluation script logs predictions vs. actuals each time it runs.")

    st.divider()

    st.subheader("Recent predictions vs. actuals")
    if len(log) > 0:
        display_log = log.tail(14)[["ds", "actual", "yhat", "abs_error"]].copy()
        display_log.columns = ["Date", "Actual (°C)", "Predicted (°C)", "Abs. Error (°C)"]
        display_log["Date"] = display_log["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display_log, width="stretch", hide_index=True)
    else:
        st.write("No prediction log yet.")

    with st.expander("About this pipeline"):
        st.write(
            """
            This dashboard reflects an automated MLOps pipeline that:

            1. **Pulls fresh weather data daily** from the Open-Meteo Historical
               Weather API for three California cities.
            2. **Retrains a Prophet forecasting model per city** on a rolling
               5-year window, with each retrain saved as a versioned artifact
               (never overwriting prior versions).
            3. **Evaluates drift** by comparing each model's predictions against
               freshly observed actuals, logging the result to a rolling
               monitoring file.
            4. **Runs entirely on a GitHub Actions schedule**, committing
               updated models and logs back to the repository automatically.
            5. **Includes retry-with-backoff** for the upstream weather API,
               since scheduled production jobs need to tolerate transient
               network failures.

            The model itself (Prophet, yearly seasonality only) is intentionally
            simple -- the point of this project is the surrounding lifecycle:
            versioning, scheduling, and monitoring, not forecasting sophistication.
            """
        )


if __name__ == "__main__":
    main()
