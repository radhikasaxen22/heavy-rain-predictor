"""Phase 1 - Step 2 (CDS variant): ERA5 daily atmospheric features.

Same output as download_era5_features.py (Open-Meteo), same JSON schema, so
build_dataset.py does not care which source produced the files:

    data/raw/era5_features/era5_<lat>_<lon>.json
    {"latitude": .., "longitude": .., "daily": {"time": [...], "temperature_2m_mean": [...]}}

Why this variant: Open-Meteo's keyless tier rate-limits long runs (observed
Retry-After of ~15 h), while Copernicus CDS has no such quota.

One-time setup:
    1. Free account: https://cds.climate.copernicus.eu
    2. Accept the licence on the dataset page:
       https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
    3. Put your key in %USERPROFILE%\\.cdsapirc :
           url: https://cds.climate.copernicus.eu/api
           key: <your-uuid>
    4. pip install cdsapi

Run:
    python src/data/download_era5_cds.py --smoke    # 1 month, 1 cell: validates setup
    python src/data/download_era5_cds.py            # all years, all selected cells

Aggregation matches Open-Meteo's daily definitions: mean/max/min 2 m temperature,
mean dew point / pressure / cloud cover, mean and max hourly wind speed, max
hourly gust, vector-mean wind direction. ERA5 hourly steps are UTC, as
Open-Meteo's default, so daily IMD dates align the same way.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BASE = Path(__file__).resolve().parents[2]
RAW = BASE / "data" / "raw" / "era5_cds"

DATASET = "reanalysis-era5-single-levels"
CDS_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "total_cloud_cover",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "instantaneous_10m_wind_gust",
]


def _load_sibling():
    """Reuse cell selection and paths from the Open-Meteo downloader: one source of truth."""
    path = Path(__file__).with_name("download_era5_features.py")
    spec = importlib.util.spec_from_file_location("_om_downloader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_om = _load_sibling()
START_YEAR = int(os.environ.get("ERA5_START", _om.START_DATE)[:4])
END_YEAR = int(os.environ.get("ERA5_END", _om.END_DATE)[:4])
ERA5_DIR = Path(_om.OUT_DIR)
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
HOURS = [f"{h:02d}:00" for h in range(24)]


def relative_humidity(temp_k, dew_k):
    """Magnus formula over water, percent. Inputs in kelvin."""

    def es(celsius):
        return 6.112 * np.exp(17.62 * celsius / (243.12 + celsius))

    t_c = temp_k - 273.15
    d_c = dew_k - 273.15
    return np.clip(100.0 * es(d_c) / es(t_c), 0.0, 100.0)


def fetch(year, area, months=None):
    """One CDS request per year -> local NetCDF. Cached; safe to re-run."""
    import cdsapi

    RAW.mkdir(parents=True, exist_ok=True)
    tag = "" if months is None else "_" + "-".join(months)
    target = RAW / f"era5_hourly_{year}{tag}.nc"
    if target.exists() and target.stat().st_size > 10_000:
        print(f"    [cache] {target.name}")
        return target

    request = {
        "product_type": ["reanalysis"],
        "variable": CDS_VARS,
        "year": [str(year)],
        "month": months or MONTHS,
        "day": DAYS,
        "time": HOURS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }
    print(f"    [cds] requesting {year} ({'all months' if months is None else months})")
    cdsapi.Client().retrieve(DATASET, request, str(target))
    print(f"    [cds] -> {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def _pick(ds, *names):
    for name in names:
        if name in ds:
            return ds[name]
    raise KeyError(f"none of {names} in {list(ds.data_vars)}")


def daily_from_hourly(ds):
    """Hourly ERA5 fields -> the daily variables Open-Meteo would have returned."""
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    time_name = "valid_time" if "valid_time" in ds.coords else "time"

    t2m = _pick(ds, "t2m", "2t")
    d2m = _pick(ds, "d2m", "2d")
    sp = _pick(ds, "sp")
    tcc = _pick(ds, "tcc")
    u10 = _pick(ds, "u10", "10u")
    v10 = _pick(ds, "v10", "10v")
    gust = _pick(ds, "i10fg", "fg10")

    speed = np.hypot(u10, v10)
    # groupby names the result dim after the index, so name it "time" here
    day = ds[time_name].dt.floor("D").rename("time")

    def daily(da, how):
        grouped = da.groupby(day)
        return getattr(grouped, how)()

    t2m_mean = daily(t2m, "mean")
    d2m_mean = daily(d2m, "mean")
    direction = (270.0 - np.degrees(np.arctan2(daily(v10, "mean"), daily(u10, "mean")))) % 360.0

    out = xr.Dataset({
        "temperature_2m_mean": t2m_mean - 273.15,
        "temperature_2m_max": daily(t2m, "max") - 273.15,
        "temperature_2m_min": daily(t2m, "min") - 273.15,
        "relative_humidity_2m_mean": relative_humidity(t2m_mean, d2m_mean),
        "dew_point_2m_mean": d2m_mean - 273.15,
        "surface_pressure_mean": daily(sp, "mean") / 100.0,
        "cloud_cover_mean": daily(tcc, "mean") * 100.0,
        "wind_speed_10m_mean": daily(speed, "mean"),
        "wind_speed_10m_max": daily(speed, "max"),
        "wind_gusts_10m_max": daily(gust, "max"),
        "wind_direction_10m_dominant": direction,
    })
    if lat_name != "latitude":
        out = out.rename({lat_name: "latitude", lon_name: "longitude"})
    return out

    if lat_name != "latitude":
        daily = daily.rename({lat_name: "latitude", lon_name: "longitude"})
    return daily


def write_cell_json(point, lat, lon):
    """point: time-indexed Dataset for a single grid cell."""
    payload = {"latitude": round(float(lat), 2), "longitude": round(float(lon), 2), "daily": {}}
    payload["daily"]["time"] = [str(pd.Timestamp(t).date()) for t in point["time"].values]
    for name in point.data_vars:
        payload["daily"][name] = [
            None if not np.isfinite(v) else round(float(v), 2) for v in point[name].values
        ]
    path = ERA5_DIR / f"era5_{lat:.2f}_{lon:.2f}.json"
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="one month, one cell: validates credentials and licence")
    args = parser.parse_args()

    box = _om.BOX
    area = [box["lat"][1], box["lon"][0], box["lat"][0], box["lon"][1]]
    cells = _om.find_land_cells()
    if args.smoke:
        cells = cells[:1]
        months, years = ["01"], [START_YEAR]
        print(f"[smoke] cell {cells[0]}, {START_YEAR}-01")
    else:
        months, years = None, list(range(START_YEAR, END_YEAR + 1))
        print(f"[plan] {len(cells)} cells x {len(years)} years ({START_YEAR}-{END_YEAR})")

    per_cell = {cell: [] for cell in cells}
    for year in years:
        with xr.open_dataset(fetch(year, area, months)) as ds:
            ds.load()
            daily = daily_from_hourly(ds)
        for lat, lon in cells:
            per_cell[(lat, lon)].append(
                daily.sel(latitude=lat, longitude=lon, method="nearest")
            )
        print(f"    {len(years) and year}: {len(cells)} cells aggregated")

    for (lat, lon), parts in per_cell.items():
        combined = xr.concat(parts, dim="time").sortby("time")
        write_cell_json(combined, lat, lon)

    files = sorted(ERA5_DIR.glob("era5_*.json"))
    print(f"[done] {len(files)} cell files in {ERA5_DIR}")
    if not args.smoke and len(files) != len(cells):
        print(f"[warn] expected {len(cells)} cells, got {len(files)}")
    if args.smoke:
        sample = files[0]
        data = json.loads(sample.read_text())["daily"]
        print("[smoke] vars:", ", ".join(k for k in data if k != "time"))
        print(f"[smoke] {len(data['time'])} days, {data['time'][0]} .. {data['time'][-1]}")


if __name__ == "__main__":
    main()
