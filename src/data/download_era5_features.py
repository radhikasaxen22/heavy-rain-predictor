"""Phase 1 - Step 2: Download ERA5 daily atmospheric features.

Source: ECMWF ERA5 reanalysis (0.25 deg) served free by Open-Meteo.
https://open-meteo.com/en/docs/historical-weather-api  (no API key needed)

We only fetch the grid cells that have real IMD land data (the Konkan
coastal strip). Each cell becomes one small JSON file in data/raw/era5_features/.
"""
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import requests
import xarray as xr

BASE = Path(__file__).resolve().parents[2]

API_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2001-01-01"
END_DATE = "2020-12-31"

DEFAULT_REGION = "maharashtra_konkan"
VALID_FRACTION = 0.9
MASK_YEARS = [2001, 2005, 2010, 2015, 2019]


def get_box():
    """Region box from data/regions.json (Maharashtra / Odisha / Kerala ...).
    Choose with the REGION environment variable, e.g.  $env:REGION='odisha_coast'"""
    region = os.environ.get("REGION", DEFAULT_REGION)
    with open(BASE / "data" / "regions.json") as fh:
        available = json.load(fh)["regions"]
    if region not in available:
        raise SystemExit(f"Unknown region '{region}'. Choose from {list(available)}")
    b = available[region]
    print(f"[region] {region} ({b['state']}) lat {b['lat']} lon {b['lon']}")
    return {"lat": (b["lat"][0], b["lat"][1]), "lon": (b["lon"][0], b["lon"][1])}, region


BOX, REGION = get_box()

RAIN_DIR = str(BASE / "data" / "raw" / "imd_rainfall")
OUT_DIR = str(BASE / "data" / "raw" / "era5_features")

DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "dew_point_2m_mean",
    "surface_pressure_mean",
    "cloud_cover_mean",
    "wind_speed_10m_mean",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
]


def find_land_cells():
    """Cells where IMD rainfall has data (>90% of days) = land cells."""
    frac = None
    for year in MASK_YEARS:
        path = os.path.join(RAIN_DIR, f"imd_rainfall_{year}.nc")
        if not os.path.exists(path):
            continue
        with xr.open_dataset(path) as ds:
            sub = ds["RAINFALL"].sel(
                LATITUDE=slice(*BOX["lat"]), LONGITUDE=slice(*BOX["lon"])
            )
            valid = ~np.isnan(sub.values)
        this_frac = valid.mean(axis=0)
        frac = this_frac if frac is None else np.maximum(frac, this_frac)
    if frac is None:
        raise FileNotFoundError("No IMD files found; run download_imd_rainfall.py first.")

    n_lat = int(round((BOX["lat"][1] - BOX["lat"][0]) / 0.25)) + 1
    n_lon = int(round((BOX["lon"][1] - BOX["lon"][0]) / 0.25)) + 1
    cells = []
    for i in range(n_lat):
        for j in range(n_lon):
            if frac[i, j] >= VALID_FRACTION:
                lat = round(BOX["lat"][0] + i * 0.25, 2)
                lon = round(BOX["lon"][0] + j * 0.25, 2)
                cells.append((lat, lon))
    return cells


def fetch_with_retry(params, max_attempts=6):
    """GET the archive API, backing off politely when rate-limited (HTTP 429)."""
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(API_URL, params=params, timeout=300)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 2 ** attempt))
                print(f"    ...rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
            else:
                print(f"    ...{exc}")
    return None


def save_cell(data):
    """Write one location's daily data to its JSON file. Return True if saved."""
    lat = round(float(data["latitude"]), 2)
    lon = round(float(data["longitude"]), 2)
    out_path = os.path.join(OUT_DIR, f"era5_{lat:.2f}_{lon:.2f}.json")
    if os.path.exists(out_path):
        return False
    daily = data.get("daily", {})
    if "time" not in daily:
        print(f"[warn] ({lat}, {lon}) no daily data: {str(data)[:150]}")
        return False
    with open(out_path, "w") as handle:
        json.dump(data, handle)
    return True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cells = find_land_cells()
    todo = [
        (lat, lon)
        for lat, lon in cells
        if not os.path.exists(os.path.join(OUT_DIR, f"era5_{lat:.2f}_{lon:.2f}.json"))
    ]
    print(f"[plan] {len(todo)}/{len(cells)} cells still to fetch")

    BATCH = 15
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        params = {
            "latitude": ",".join(str(c[0]) for c in batch),
            "longitude": ",".join(str(c[1]) for c in batch),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "daily": ",".join(DAILY_VARS),
            "timezone": "Asia/Kolkata",
            "models": "era5",
        }
        print(f"[fetch batch] {i + 1}-{i + len(batch)}/{len(todo)} ({len(batch)} cells) ...")
        data = fetch_with_retry(params)
        if data is None:
            print(f"[error] batch {i // BATCH} gave up after retries")
            continue
        items = data if isinstance(data, list) else [data]
        saved = sum(1 for item in items if save_cell(item))
        print(f"[ok] saved {saved} cells from batch")
        time.sleep(2)

    print("[done] ERA5 features downloaded.")


if __name__ == "__main__":
    main()
