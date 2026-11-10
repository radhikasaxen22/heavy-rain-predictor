"""Feature registry - the single source of truth for model features.

Every meteorology + derived feature used by the model is listed here with its
units, data source and physical reason for inclusion.  To add a future feature
(e.g. an INSAT-3D/3DR brightness-temperature column) you only add one dict here;
nothing else in the pipeline changes.
"""

FEATURES = [
    {"name": "temp_mean", "unit": "degC", "source": "ERA5",
     "why": "Air temperature; warm air holds more moisture available to rain out."},
    {"name": "temp_max", "unit": "degC", "source": "ERA5",
     "why": "Daily peak heating drives afternoon convection and storm growth."},
    {"name": "temp_min", "unit": "degC", "source": "ERA5",
     "why": "Night-time minimum relates to humidity build-up and stability."},
    {"name": "relative_humidity", "unit": "%", "source": "ERA5",
     "why": "Near-surface moisture - the direct fuel for heavy rainfall."},
    {"name": "dew_point", "unit": "degC", "source": "ERA5",
     "why": "Absolute moisture content of the air, independent of heating."},
    {"name": "surface_pressure", "unit": "hPa", "source": "ERA5",
     "why": "Low pressure means converging, rising air that produces clouds/rain."},
    {"name": "cloud_cover", "unit": "%", "source": "ERA5",
     "why": "Pre-monsoon/convective cloud build-up precedes heavy rain."},
    {"name": "wind_speed_mean", "unit": "km/h", "source": "ERA5",
     "why": "Transports moist ocean air into the region (monsoon flow)."},
    {"name": "wind_speed_max", "unit": "km/h", "source": "ERA5",
     "why": "Peak wind bursts signal active convective (storm) cells."},
    {"name": "wind_gust", "unit": "km/h", "source": "ERA5",
     "why": "Gustiness is a marker of convective (storm) activity."},
    {"name": "wind_direction", "unit": "deg", "source": "ERA5",
     "why": "South-westerly flow = monsoon moisture push; land/sea flow matters."},
    {"name": "rain_lag1", "unit": "mm", "source": "Derived (IMD prev. day)",
     "why": "Persistence - a wet day is often followed by another wet day."},
    {"name": "day_of_year", "unit": "day", "source": "Derived",
     "why": "Captures monsoon seasonality (June-September peak)."},
]

MODEL_FEATURES = [f["name"] for f in FEATURES]

# --- Circular features -------------------------------------------------------
# wind_direction and day_of_year are angles on a circle: 359 deg is one step
# from 1 deg, and day 365 is one step from day 1. Treated as plain numbers
# (what the committed models do) those neighbouring pairs look maximally far
# apart, both to the model and to the OOD z-score check in reliability.py.
#
# CIRCULAR_ENCODING is OFF because models/xgb_ens_*.json were trained on the
# linear features and expect exactly MODEL_FEATURES in that order. Flipping it
# adds two columns, so it REQUIRES a retrain (train_model.py + build_shap.py)
# and a fresh metadata.json before the API can serve it.
CIRCULAR_ENCODING = False
CIRCULAR_FEATURES = {"wind_direction": 360.0, "day_of_year": 365.0}


def circular_terms(name, value):
    """Return (sin, cos) for a circular feature, or None if it is not circular."""
    import math

    period = CIRCULAR_FEATURES.get(name)
    if period is None or value is None:
        return None
    angle = 2.0 * math.pi * float(value) / period
    return math.sin(angle), math.cos(angle)


def active_features():
    """Feature list the model is trained/served with, honouring CIRCULAR_ENCODING."""
    if not CIRCULAR_ENCODING:
        return list(MODEL_FEATURES)
    out = []
    for name in MODEL_FEATURES:
        if name in CIRCULAR_FEATURES:
            out += [f"{name}_sin", f"{name}_cos"]
        else:
            out.append(name)
    return out


def encode_row(row, names=None):
    """Return a new row using the active feature names."""
    names = names or active_features()
    out = {}
    for name in names:
        if name.endswith("_sin") or name.endswith("_cos"):
            base, part = name.rsplit("_", 1)
            terms = circular_terms(base, row.get(base))
            if terms is None:
                continue
            out[name] = terms[0] if part == "sin" else terms[1]
        else:
            out[name] = row.get(name)
    return out

# metadata columns carried along for analysis / spatial reference (NOT model inputs)
META_COLUMNS = ["date", "latitude", "longitude", "rainfall_mm"]

TARGET = "target"
RAIN_COL = "rainfall_mm"
THRESHOLD_MM = 64.5  # IMD "Very Heavy Rain" lower bound
