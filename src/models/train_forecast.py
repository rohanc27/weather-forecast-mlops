"""Train a Prophet forecasting model for daily max temperature, per city.

Each city gets its own model. Versioned saves: every training run creates
a new timestamped folder under models/, and updates a JSON registry
tracking all versions and their validation metrics.

Usage:
    python -m src.models.train_forecast --city mountain_view
    python -m src.models.train_forecast --city all
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"

CITIES = ["mountain_view", "san_francisco", "los_angeles"]

# Holdout the most recent N days as a validation set
VALIDATION_DAYS = 30


def load_city_data(city: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW_DIR / f"{city}.parquet")
    # Prophet requires columns named exactly 'ds' (date) and 'y' (target)
    df = df.rename(columns={"date": "ds", "temp_max_c": "y"})
    return df[["ds", "y"]].sort_values("ds").reset_index(drop=True)


def train_one_city(city: str) -> dict:
    print(f"\n=== Training {city} ===")
    df = load_city_data(city)
    print(f"  Loaded {len(df):,} rows ({df['ds'].min().date()} to {df['ds'].max().date()})")

    train_df = df.iloc[:-VALIDATION_DAYS]
    val_df = df.iloc[-VALIDATION_DAYS:]
    print(f"  Train: {len(train_df):,} rows, Validation: {len(val_df):,} rows (most recent {VALIDATION_DAYS} days)")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,  # weather has no weekly pattern
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )

    print("  Fitting Prophet model...")
    model.fit(train_df)

    # Validate: predict the held-out days, compare to actual
    future = val_df[["ds"]].copy()
    forecast = model.predict(future)

    mae = mean_absolute_error(val_df["y"], forecast["yhat"])
    print(f"  Validation MAE: {mae:.2f}°C")

    # Save versioned model
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_dir = MODELS_DIR / city / f"v_{timestamp}"
    version_dir.mkdir(parents=True, exist_ok=True)
    model_path = version_dir / "model.joblib"
    joblib.dump(model, model_path)
    print(f"  Saved model to {model_path}")

    return {
        "city": city,
        "version": timestamp,
        "trained_at": timestamp,
        "train_rows": len(train_df),
        "validation_mae_c": round(float(mae), 3),
        "data_start": str(df["ds"].min().date()),
        "data_end": str(df["ds"].max().date()),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
    }


def update_registry(new_entries: list[dict]) -> None:
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text())
    else:
        registry = {"models": []}

    registry["models"].extend(new_entries)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
    print(f"\nRegistry updated: {REGISTRY_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=CITIES + ["all"], default="mountain_view")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cities_to_train = CITIES if args.city == "all" else [args.city]

    results = []
    for city in cities_to_train:
        result = train_one_city(city)
        results.append(result)

    update_registry(results)

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['city']}: MAE = {r['validation_mae_c']}°C, version = {r['version']}")


if __name__ == "__main__":
    main()
