"""Evaluate forecast accuracy by comparing past predictions to actuals.

For each city, finds the most recent model version, generates a forecast
for a recent window, compares it against the actual observed temperatures
(pulled fresh), and appends results to a rolling monitoring log.

This is the core "model health" signal: as new actual data arrives daily,
we can see whether the live model's predictions are still accurate, or
whether performance has drifted and a retrain/investigation is warranted.

Usage:
    python -m src.monitoring.evaluate_drift
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
MONITORING_DIR = PROJECT_ROOT / "monitoring"
REGISTRY_PATH = MODELS_DIR / "registry.json"

CITIES = {
    "mountain_view": {"lat": 37.3861, "lon": -122.0839, "tz": "America/Los_Angeles"},
    "san_francisco": {"lat": 37.7749, "lon": -122.4194, "tz": "America/Los_Angeles"},
    "los_angeles": {"lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles"},
}

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Flag drift if rolling 7-day MAE exceeds this multiple of the model's
# original validation MAE
DRIFT_THRESHOLD_MULTIPLIER = 1.5


def get_latest_model_path(city: str) -> Path:
    """Find the most recently trained model version for a city."""
    city_dir = MODELS_DIR / city
    versions = sorted([d for d in city_dir.iterdir() if d.is_dir() and d.name.startswith("v_")])
    if not versions:
        raise FileNotFoundError(f"No trained models found for {city}")
    return versions[-1] / "model.joblib"


def get_latest_validation_mae(city: str) -> float:
    """Pull this city's most recent validation MAE from the registry."""
    registry = json.loads(REGISTRY_PATH.read_text())
    city_entries = [m for m in registry["models"] if m["city"] == city]
    if not city_entries:
        raise ValueError(f"No registry entries for {city}")
    return city_entries[-1]["validation_mae_c"]


def fetch_recent_actuals(city: str, coords: dict, days: int = 14) -> pd.DataFrame:
    """Pull the most recent N days of actual observed temperatures."""
    from datetime import date, timedelta
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=days)

    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_max",
        "timezone": coords["tz"],
    }
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    return pd.DataFrame({
        "ds": pd.to_datetime(data["daily"]["time"]),
        "actual": data["daily"]["temperature_2m_max"],
    })


def evaluate_city(city: str, coords: dict) -> dict:
    print(f"\n=== Evaluating {city} ===")

    model_path = get_latest_model_path(city)
    model = joblib.load(model_path)
    baseline_mae = get_latest_validation_mae(city)
    print(f"  Using model: {model_path.parent.name} (baseline validation MAE: {baseline_mae}°C)")

    actuals = fetch_recent_actuals(city, coords)
    print(f"  Fetched {len(actuals)} days of recent actuals")

    forecast = model.predict(actuals[["ds"]])
    merged = actuals.merge(forecast[["ds", "yhat"]], on="ds")
    merged["abs_error"] = (merged["actual"] - merged["yhat"]).abs()

    rolling_mae = merged["abs_error"].mean()
    drift_flag = rolling_mae > (baseline_mae * DRIFT_THRESHOLD_MULTIPLIER)

    print(f"  Rolling {len(merged)}-day MAE: {rolling_mae:.2f}°C")
    print(f"  Drift threshold ({DRIFT_THRESHOLD_MULTIPLIER}x baseline): {baseline_mae * DRIFT_THRESHOLD_MULTIPLIER:.2f}°C")
    print(f"  Drift detected: {drift_flag}")

    log_path = MONITORING_DIR / f"{city}_predictions.csv"
    merged_out = merged.copy()
    merged_out["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    merged_out["model_version"] = model_path.parent.name

    if log_path.exists():
        existing = pd.read_csv(log_path, parse_dates=["ds"])
        combined = pd.concat([existing, merged_out], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ds"], keep="last")
    else:
        combined = merged_out

    combined = combined.sort_values("ds")
    combined.to_csv(log_path, index=False)
    print(f"  Log updated: {log_path}")

    return {
        "city": city,
        "rolling_mae": round(float(rolling_mae), 3),
        "baseline_mae": baseline_mae,
        "drift_detected": bool(drift_flag),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=list(CITIES.keys()) + ["all"], default="all")
    args = parser.parse_args()

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    cities_to_eval = CITIES if args.city == "all" else {args.city: CITIES[args.city]}

    results = []
    for city, coords in cities_to_eval.items():
        results.append(evaluate_city(city, coords))

    print("\n=== Drift summary ===")
    for r in results:
        status = "⚠ DRIFT" if r["drift_detected"] else "OK"
        print(f"  {r['city']}: rolling MAE = {r['rolling_mae']}°C, "
              f"baseline = {r['baseline_mae']}°C [{status}]")


if __name__ == "__main__":
    main()
