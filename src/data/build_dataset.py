"""Phase 1 - Step 3: Merge IMD rainfall (target) + ERA5 atmosphere (features).

Output: data/processed/rainfall_dataset.csv
One row per (date, grid cell) on the Konkan coast, with:
  - features: temperature, humidity, dew point, pressure, cloud, wind, rain_lag1, season
  - target: 1 if IMD daily rainfall > 64.5 mm (heavy rain), else 0
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BASE = Path(__file__).resolve().parents[2]
RAIN_DIR = str(BASE / "data" / "raw" / "imd_rainfall")
ERA5_DIR = str(BASE / "data" / "raw" / "era5_features")

THRESHOLD_MM = 64.5
DEFAULT_REGION = "maharashtra_konkan"


def get_box():
    region = os.environ.get("REGION", DEFAULT_REGION)
    with open(BASE / "data" / "regions.json") as fh:
        available = json.load(fh)["regions"]
    if region not in available:
        raise SystemExit(f"Unknown region '{region}'. Choose from {list(available)}")
    b = available[region]
    print(f"[region] {region} ({b['state']})")
    return {"lat": (b["lat"][0], b["lat"][1]), "lon": (b["lon"][0], b["lon"][1])}, region


BOX, REGION = get_box()
OUT_CSV = str(BASE / "data" / "processed" / (f"rainfall_dataset.csv"
             if REGION == "maharashtra_konkan" else f"rainfall_dataset_{REGION}.csv"))

FEATURE_RENAME = {
    "temperature_2m_mean": "temp_mean",
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "relative_humidity_2m_mean": "relative_humidity",
    "dew_point_2m_mean": "dew_point",
    "surface_pressure_mean": "surface_pressure",
    "cloud_cover_mean": "cloud_cover",
    "wind_speed_10m_mean": "wind_speed_mean",
    "wind_speed_10m_max": "wind_speed_max",
    "wind_gusts_10m_max": "wind_gust",
    "wind_direction_10m_dominant": "wind_direction",
}


def load_imd():
    frames = []
    for path in sorted(glob.glob(os.path.join(RAIN_DIR, "imd_rainfall_*.nc"))):
        with xr.open_dataset(path) as ds:
            sub = ds["RAINFALL"].sel(
                LATITUDE=slice(*BOX["lat"]), LONGITUDE=slice(*BOX["lon"])
            )
            df = sub.to_dataframe(name="rainfall_mm").reset_index()
        df["date"] = pd.to_datetime(df["TIME"]).dt.date
        df = df.rename(
            columns={"LATITUDE": "latitude", "LONGITUDE": "longitude"}
        )[["date", "latitude", "longitude", "rainfall_mm"]]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_era5():
    frames = []
    for path in glob.glob(os.path.join(ERA5_DIR, "era5_*.json")):
        with open(path) as handle:
            data = json.load(handle)
        lat = round(float(data["latitude"]), 2)
        lon = round(float(data["longitude"]), 2)
        df = pd.DataFrame(data["daily"])
        df = df.rename(columns=FEATURE_RENAME)
        df["date"] = pd.to_datetime(df["time"]).dt.date
        df = df.drop(columns=["time"])
        df["latitude"] = lat
        df["longitude"] = lon
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    print("[1/4] loading ERA5 features ...")
    era5 = load_era5()
    era5["latitude"] = era5["latitude"].round(2)
    era5["longitude"] = era5["longitude"].round(2)

    print("[2/4] loading IMD rainfall target ...")
    imd = load_imd()
    imd["latitude"] = imd["latitude"].round(2)
    imd["longitude"] = imd["longitude"].round(2)

    print("[3/4] merging on (date, latitude, longitude) ...")
    df = imd.merge(era5, on=["date", "latitude", "longitude"], how="inner")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["latitude", "longitude", "date"]).reset_index(drop=True)

    print("[4/4] feature engineering ...")
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["rain_lag1"] = df.groupby(["latitude", "longitude"])["rainfall_mm"].shift(1)
    df["target"] = (df["rainfall_mm"] > THRESHOLD_MM).astype(int)

    n_before = len(df)
    df = df.dropna()
    print(f"    dropped {n_before - len(df)} rows with missing values")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"[saved] {OUT_CSV} ({len(df):,} rows)")
    print(f"    date range   : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"    grid cells   : {df[['latitude', 'longitude']].drop_duplicates().shape[0]}")
    print(f"    heavy-rain % : {df['target'].mean() * 100:.2f}%  ({df['target'].sum():,} events)")


if __name__ == "__main__":
    main()