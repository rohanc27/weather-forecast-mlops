"""Pull historical daily max temperature for three CA cities via Open-Meteo.

Uses the Historical Weather API (ERA5 reanalysis) -- no API key required.
Pulls the last N years of daily data per city, saved as separate parquet
files so each city's model can be trained independently.

Usage:
    python -m src.data.pull_weather
    python -m src.data.pull_weather --years 5
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

CITIES = {
    "mountain_view": {"lat": 37.3861, "lon": -122.0839, "tz": "America/Los_Angeles"},
    "san_francisco": {"lat": 37.7749, "lon": -122.4194, "tz": "America/Los_Angeles"},
    "los_angeles": {"lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles"},
}

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

DEFAULT_YEARS = 5


def pull_city(city_name: str, lat: float, lon: float, tz: str, years: int) -> pd.DataFrame:
    """Pull daily max temp for one city, last `years` years up to yesterday."""
    end_date = date.today() - timedelta(days=1)  # yesterday (today's data incomplete)
    start_date = end_date.replace(year=end_date.year - years)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_max",
        "timezone": tz,
    }

    print(f"  Pulling {city_name}: {start_date} to {end_date}...")
    data = None
    last_exc = None
    for attempt in range(1, 5):
        try:
            response = requests.get(BASE_URL, params=params, timeout=45)
            response.raise_for_status()
            data = response.json()
            break
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            wait = min(2 ** attempt, 30)
            print(f"    Attempt {attempt}/4 failed ({type(exc).__name__}), retrying in {wait}s...")
            time.sleep(wait)
    if data is None:
        raise RuntimeError(f"Failed to pull {city_name} after 4 attempts") from last_exc

    df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "temp_max_c": data["daily"]["temperature_2m_max"],
    })
    df["city"] = city_name
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for city_name, coords in CITIES.items():
        out_path = RAW_DIR / f"{city_name}.parquet"

        if out_path.exists() and not args.force:
            existing = pd.read_parquet(out_path)
            print(f"Already pulled: {city_name} ({len(existing):,} rows, "
                  f"{existing['date'].min().date()} to {existing['date'].max().date()})")
            continue

        df = pull_city(city_name, coords["lat"], coords["lon"], coords["tz"], args.years)
        df.to_parquet(out_path, index=False)
        print(f"  Saved {len(df):,} rows to {out_path.name} "
              f"(missing values: {df['temp_max_c'].isna().sum()})")
        time.sleep(1.0)  # polite delay between requests

    print("\nDone. Per-city summary:")
    for city_name in CITIES:
        df = pd.read_parquet(RAW_DIR / f"{city_name}.parquet")
        print(f"  {city_name}: {len(df):,} rows, "
              f"mean {df['temp_max_c'].mean():.1f}°C, "
              f"range [{df['temp_max_c'].min():.1f}, {df['temp_max_c'].max():.1f}]°C")


if __name__ == "__main__":
    main()
