"""Phase 1 - Step 2: Download ERA5 daily atmospheric features.

Source: ECMWF ERA5 reanalysis (0.25 deg) served free by Open-Meteo.
https://open-meteo.com/en/docs/historical-weather-api  (no API key needed)

We only fetch the grid cells that have real IMD land data (the Konkan
coastal strip). Each cell becomes one small JSON file in data/raw/era5_features/.

Free-tier scope: Open-Meteo refuses a request covering 15 cells x 11 variables x
20 years as "too much data", and the keyless tier rate-limits long runs. The
defaults below are the practical free scope (2011-2020, every 2nd cell). Widen
them with ERA5_START, ERA5_END and ERA5_CELL_STRIDE once you have a paid key;
build_dataset.py merges with an inner join, so it adapts to whatever is present.
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
START_DATE = os.environ.get("ERA5_START", "2011-01-01")
END_DATE = os.environ.get("ERA5_END", "2020-12-31")
CELL_STRIDE = max(1, int(os.environ.get("ERA5_CELL_STRIDE", "2")))

DEFAULT_REGION = "maharashtra_konkan"
VALID_FRACTION = 0.9
MASK_YEARS = [2001, 2005, 2010, 2015, 2019]

# Open-Meteo rejects a request as "too much data" above roughly this many
# (locations x variables x days) units; 15 cells x 11 variables was refused,
# 15 x 4 was accepted. Stay well under it and split further if still refused.
COST_BUDGET = 40
PAUSE_SECONDS = 8

_TOO_BIG = object()


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
            if i % CELL_STRIDE or j % CELL_STRIDE:
                continue
            if frac[i, j] >= VALID_FRACTION:
                lat = round(BOX["lat"][0] + i * 0.25, 2)
                lon = round(BOX["lon"][0] + j * 0.25, 2)
                cells.append((lat, lon))
    return cells


def fetch_with_retry(params, max_attempts=6, max_rate_limit_waits=12):
    """GET the archive API, honouring Retry-After when rate-limited.

    A 429 can ask for 20+ minutes of cooldown. Counting that as a failed attempt
    gives up after a handful of hours of waiting, so cooldowns are tracked
    separately from genuine failures.
    """
    failures = 0
    cooldowns = 0
    while failures < max_attempts:
        try:
            resp = requests.get(API_URL, params=params, timeout=300)
            if resp.status_code == 429:
                cooldowns += 1
                if cooldowns > max_rate_limit_waits:
                    print("    ...too many rate-limit cooldowns, giving up")
                    return None
                wait = int(resp.headers.get("Retry-After", 60))
                if os.environ.get("ERA5_EXIT_ON_RATE_LIMIT"):
                    # Caller retries later (e.g. a scheduler); do not sit in a
                    # 14-hour sleep that a login session or job object can kill.
                    print(f"    ...rate limited for {wait}s, exiting so a later run can retry")
                    raise SystemExit(3)
                print(f"    ...rate limited, waiting {wait}s")
                time.sleep(wait + 1)
                continue
            if resp.status_code == 400 and "too much data" in resp.text:
                return _TOO_BIG
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            failures += 1
            if failures < max_attempts:
                time.sleep(2 ** failures)
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


def fetch_cells(cells):
    """Fetch one batch; halve it if the API still says the request is too large."""
    params = {
        "latitude": ",".join(str(c[0]) for c in cells),
        "longitude": ",".join(str(c[1]) for c in cells),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Asia/Kolkata",
        "models": "era5",
    }
    data = fetch_with_retry(params)
    if data is _TOO_BIG:
        if len(cells) == 1:
            print(f"[error] single cell still too large: {cells[0]}")
            return []
        mid = len(cells) // 2
        print(f"    ...too much data, splitting {len(cells)} cells into {mid}+{len(cells) - mid}")
        return fetch_cells(cells[:mid]) + fetch_cells(cells[mid:])
    if data is None:
        return []
    return data if isinstance(data, list) else [data]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cells = find_land_cells()
    todo = [
        (lat, lon)
        for lat, lon in cells
        if not os.path.exists(os.path.join(OUT_DIR, f"era5_{lat:.2f}_{lon:.2f}.json"))
    ]
    batch_size = max(1, COST_BUDGET // len(DAILY_VARS))
    print(f"[plan] {len(todo)}/{len(cells)} cells still to fetch, {batch_size} per request")

    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        print(f"[fetch batch] {i + 1}-{i + len(batch)}/{len(todo)} ({len(batch)} cells) ...")
        items = fetch_cells(batch)
        saved = sum(1 for item in items if save_cell(item))
        print(f"[ok] saved {saved}/{len(batch)} cells from batch")
        time.sleep(PAUSE_SECONDS)

    print("[done] ERA5 features downloaded.")


if __name__ == "__main__":
    main()
