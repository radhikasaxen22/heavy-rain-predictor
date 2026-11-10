"""Offline checks on the data pipeline's contracts.

These do not touch the network and do not need the downloaded NetCDF/JSON
corpora except where explicitly skipped.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

BASE = Path(__file__).resolve().parents[1]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, BASE / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def _hourly_day():
    hours = pd.date_range("2015-06-01", periods=24, freq="h")
    shape = (24, 1, 1)

    def const(value):
        return xr.DataArray(
            np.full(shape, float(value)),
            dims=("valid_time", "latitude", "longitude"),
            coords={"valid_time": hours, "latitude": [17.0], "longitude": [73.0]},
        )

    return xr.Dataset(
        {
            "t2m": const(300.15),
            "d2m": const(295.15),
            "sp": const(101300.0),
            "tcc": const(0.5),
            "u10": const(3.0),
            "v10": const(4.0),
            "i10fg": const(12.0),
        }
    )


@pytest.fixture(scope="module")
def cds():
    return _load("era5_cds_under_test", "src/data/download_era5_cds.py")


def test_daily_aggregation_units_and_conventions(cds):
    daily = cds.daily_from_hourly(_hourly_day())

    assert daily.sizes["time"] == 1, "24 hourly steps must collapse to one day"
    row = {name: float(daily[name].values[0, 0, 0]) for name in daily.data_vars}

    assert row["temperature_2m_mean"] == pytest.approx(27.0)      # K -> degC
    assert row["temperature_2m_max"] == pytest.approx(27.0)
    assert row["temperature_2m_min"] == pytest.approx(27.0)
    assert row["dew_point_2m_mean"] == pytest.approx(22.0)
    assert row["surface_pressure_mean"] == pytest.approx(1013.0)  # Pa -> hPa
    assert row["cloud_cover_mean"] == pytest.approx(50.0)         # fraction -> %
    assert row["wind_speed_10m_mean"] == pytest.approx(5.0)       # hypot(3, 4)
    assert row["wind_speed_10m_max"] == pytest.approx(5.0)
    assert row["wind_gusts_10m_max"] == pytest.approx(12.0)       # passthrough

    def magnus(kelvin):
        celsius = kelvin - 273.15
        return 6.112 * np.exp(17.62 * celsius / (243.12 + celsius))

    expected_rh = 100.0 * magnus(295.15) / magnus(300.15)
    assert row["relative_humidity_2m_mean"] == pytest.approx(expected_rh, abs=1e-9)

    # meteorological convention: wind FROM the south-west for u=3 (east), v=4 (north)
    expected_direction = (270.0 - np.degrees(np.arctan2(4.0, 3.0))) % 360.0
    assert row["wind_direction_10m_dominant"] == pytest.approx(expected_direction, abs=1e-9)
    assert 0.0 <= row["wind_direction_10m_dominant"] < 360.0


def test_land_mask_and_sampling_stride():
    om = _load("era5_om_under_test", "src/data/download_era5_features.py")

    om.CELL_STRIDE = 1
    all_cells = om.find_land_cells()
    assert len(all_cells) == 131

    om.CELL_STRIDE = 2
    sampled = om.find_land_cells()
    assert len(sampled) == 36
    assert set(sampled) <= set(all_cells)


def test_sampled_cells_sit_on_imd_grid():
    om = _load("era5_om_grid_test", "src/data/download_era5_features.py")
    imd_files = sorted((BASE / "data" / "raw" / "imd_rainfall").glob("imd_rainfall_*.nc"))
    if not imd_files:
        pytest.skip("IMD NetCDF files not downloaded")

    with xr.open_dataset(imd_files[0]) as ds:
        lats = {round(float(v), 2) for v in ds["LATITUDE"].values}
        lons = {round(float(v), 2) for v in ds["LONGITUDE"].values}

    off_grid = [c for c in om.find_land_cells()
                if round(c[0], 2) not in lats or round(c[1], 2) not in lons]
    assert not off_grid, f"cells missing from the IMD grid: {off_grid[:5]}"
