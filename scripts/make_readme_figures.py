"""Generate charts for the README: error trends per city, model comparison,
and the pipeline architecture diagram.

Usage:
    python scripts/make_readme_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MONITORING_DIR = PROJECT_ROOT / "monitoring"
FIGURES_DIR = PROJECT_ROOT / "figures"

CITIES = ["mountain_view", "san_francisco", "los_angeles"]
CITY_LABELS = {
    "mountain_view": "Mountain View",
    "san_francisco": "San Francisco",
    "los_angeles": "Los Angeles",
}
COLORS = {"mountain_view": "#3b82f6", "san_francisco": "#10b981", "los_angeles": "#f59e0b"}

DRIFT_THRESHOLD_MULTIPLIER = 1.5


def fig_error_trends():
    """One combined chart: rolling 7-day MAE per city over time."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for city in CITIES:
        path = MONITORING_DIR / f"{city}_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["ds"]).sort_values("ds")
        rolling = df["abs_error"].rolling(window=min(7, len(df)), min_periods=1).mean()
        ax.plot(df["ds"], rolling, marker="o", markersize=3, color=COLORS[city],
                label=f"{CITY_LABELS[city]} (7-day avg)")

    ax.set_ylabel("Absolute error (°C)", fontsize=11)
    ax.set_title("Rolling prediction error across all three cities", fontsize=13, weight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_trends.png", dpi=200)
    plt.close(fig)
    print("Saved error_trends.png")


def fig_model_comparison():
    """Bar chart: training MAE per city, most recent version."""
    registry = json.loads((MODELS_DIR / "registry.json").read_text())

    latest_per_city = {}
    for entry in registry["models"]:
        latest_per_city[entry["city"]] = entry  # keeps overwriting, last one wins

    cities = [c for c in CITIES if c in latest_per_city]
    maes = [latest_per_city[c]["validation_mae_c"] for c in cities]
    labels = [CITY_LABELS[c] for c in cities]
    colors = [COLORS[c] for c in cities]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, maes, color=colors, width=0.5)
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, mae + 0.05, f"{mae:.2f}°C",
                 ha="center", fontsize=11, weight="bold")

    ax.set_ylabel("Validation MAE (°C)", fontsize=11)
    ax.set_title("Most recent model accuracy by city", fontsize=13, weight="bold")
    ax.set_ylim(0, max(maes) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=200)
    plt.close(fig)
    print("Saved model_comparison.png")


def fig_version_count():
    """Bar chart: how many retrains have accumulated per city."""
    registry = json.loads((MODELS_DIR / "registry.json").read_text())

    counts = {city: 0 for city in CITIES}
    for entry in registry["models"]:
        if entry["city"] in counts:
            counts[entry["city"]] += 1

    labels = [CITY_LABELS[c] for c in CITIES]
    values = [counts[c] for c in CITIES]
    colors = [COLORS[c] for c in CITIES]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, str(v),
                 ha="center", fontsize=12, weight="bold")

    ax.set_ylabel("Model versions retained", fontsize=11)
    ax.set_title("Versioned retrains per city", fontsize=13, weight="bold")
    ax.set_ylim(0, max(values) + 1)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "version_count.png", dpi=200)
    plt.close(fig)
    print("Saved version_count.png")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_error_trends()
    fig_model_comparison()
    fig_version_count()
    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()