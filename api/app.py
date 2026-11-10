"""Phase 6: FastAPI backend for the heavy-rainfall predictor.

Endpoints:
  POST /predict  - weather input -> prediction + probability + reliability + XAI
  GET  /forecast - live prediction for a city (Open-Meteo, no API key)
  GET  /metrics  - model performance + global SHAP importance
  GET  /features - feature definitions
  GET  /regions  - region/city registry
  GET  /         - the web dashboard (static frontend)

All model/asset paths are resolved from this file, never from the current
working directory, so the service behaves identically under uvicorn, under
pytest, and from any shell location.
"""
import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np
import requests
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

BASE = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE / "models"
sys.path.insert(0, str(BASE / "src"))

from features.feature_config import (  # noqa: E402
    CIRCULAR_ENCODING, FEATURES, MODEL_FEATURES, encode_row,
)
from reliability.reliability import assess, load_metadata  # noqa: E402
from xai.explain import explain_local  # noqa: E402

app = FastAPI(title="Heavy Rainfall Predictor", version="0.2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# --------------------------------------------------------------------------
# Assets are loaded once at import, never per request.
# --------------------------------------------------------------------------
N_MEMBERS = 3


def _load_models():
    members = []
    for i in range(N_MEMBERS):
        m = xgb.XGBClassifier()
        m.load_model(str(MODELS_DIR / f"xgb_ens_{i}.json"))
        members.append(m)
    return members


def _load_json(name):
    with open(MODELS_DIR / name) as fh:
        return json.load(fh)


def _load_calibrator():
    """Optional isotonic calibration knots written by train_model.py.

    Absent for models trained before calibration existed, in which case raw
    ensemble probabilities are served unchanged.
    """
    path = MODELS_DIR / "calibration.json"
    if not path.exists():
        return None
    knots = json.loads(path.read_text())
    x = np.asarray(knots.get("x", []), dtype=float)
    y = np.asarray(knots.get("y", []), dtype=float)
    if x.size < 2 or x.size != y.size:
        return None
    return x, y


_model_members = _load_models()
_metadata = load_metadata()
_metrics = _load_json("metrics.json")
_global_shap = _load_json("global_shap.json")
_regions = json.loads((BASE / "data" / "regions.json").read_text())
_calibrator = _load_calibrator()

FEATURE_INFO = {f["name"]: f for f in FEATURES}
# Threshold chosen during training (validation set); 0.5 for older artefacts.
THRESHOLD = float(_metrics.get("threshold_prediction", 0.5))


def _disclaimer():
    """State the data window the served model actually saw, not a stale constant."""
    start = _metadata.get("date_min")
    end = _metadata.get("date_max")
    if start and end:
        window = f"{str(start)[:4]}-{str(end)[:4]}"
    else:
        window = "public IMD/ERA5"
    return (f"Prototype built on public IMD/ERA5 data ({window}, Konkan coast). "
            "Not an operational ISRO forecasting system.")

WMO_CODES = {0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
             45: "Fog", 48: "Fog", 51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
             61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain",
             67: "Freezing rain", 71: "Slight snow", 73: "Snow", 75: "Heavy snow",
             80: "Rain showers", 81: "Rain showers", 82: "Violent showers",
             95: "Thunderstorm", 96: "Thunderstorm + hail", 99: "Thunderstorm + hail"}


class WeatherInput(BaseModel):
    """One day of weather. Strict types reject bools and strings as numbers."""

    model_config = ConfigDict(strict=True, extra="forbid")

    temp_mean: float = Field(..., ge=-30, le=55,
                             description="Daily mean air temperature (degC)")
    temp_max: float = Field(..., ge=-30, le=60, description="Daily max temperature (degC)")
    temp_min: float = Field(..., ge=-40, le=50, description="Daily min temperature (degC)")
    relative_humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    dew_point: float = Field(..., ge=-50, le=50, description="Dew point (degC)")
    surface_pressure: float = Field(..., ge=800, le=1100, description="Surface pressure (hPa)")
    cloud_cover: float = Field(..., ge=0, le=100, description="Cloud cover (%)")
    wind_speed_mean: float = Field(..., ge=0, le=250, description="Mean wind speed (km/h)")
    wind_speed_max: float = Field(..., ge=0, le=350, description="Max wind speed (km/h)")
    wind_gust: float = Field(..., ge=0, le=350, description="Wind gust (km/h)")
    wind_direction: float = Field(..., ge=0, le=360, description="Wind direction (deg)")
    rain_lag1: float = Field(..., ge=0, le=2000, description="Previous day rainfall (mm)")
    day_of_year: float = Field(..., ge=1, le=366, description="Day of year (1-366)")


def dew_point(temp_c, rh):
    """Magnus formula: dew point from air temperature and relative humidity."""
    a, b = 17.27, 237.7
    gamma = (a * temp_c) / (b + temp_c) + math.log(rh / 100.0)
    return (b * gamma) / (a - gamma)


def _calibrate(p):
    if _calibrator is None:
        return float(p)
    x, y = _calibrator
    return float(np.clip(np.interp(p, x, y), 0.0, 1.0))


def _predict_proba(row):
    """Ensemble mean probability (optionally calibrated) + member disagreement."""
    names = _metadata.get("model_features", MODEL_FEATURES)
    X = np.array([[row[f] for f in names]], dtype=float)
    probs = np.array([m.predict_proba(X)[:, 1][0] for m in _model_members])
    return _calibrate(float(np.mean(probs))), float(np.std(probs))


def _build_explanation(row, limit=8):
    """Shared by /predict and /forecast. Feature metadata comes from the registry."""
    out = []
    for e in explain_local(row)[:limit]:
        info = FEATURE_INFO.get(e["feature"], {})
        out.append({
            "feature": e["feature"],
            "unit": info.get("unit"),
            "value": e["value"],
            "shap": e["shap"],
            "direction": e["direction"],
            "impact": e["impact"],
        })
    return out


def _risk_level(prob):
    return "HIGH" if prob >= 0.65 else ("MEDIUM" if prob >= 0.35 else "LOW")


@app.get("/")
def root():
    return FileResponse(BASE / "frontend" / "index.html")


@app.get("/features")
def features():
    return {"model_features": _metadata.get("model_features", MODEL_FEATURES),
            "circular_encoding": CIRCULAR_ENCODING,
            "features": FEATURES}


@app.get("/regions")
def regions():
    return _regions


@app.get("/metrics")
def metrics():
    return {"metrics": _metrics, "global_shap": _global_shap}


# --------------------------------------------------------------------------
# Live forecast
# --------------------------------------------------------------------------
DAILY_VARS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
              "wind_speed_10m_max", "precipitation_sum"]
CURRENT_VARS = ["relative_humidity_2m", "cloud_cover", "surface_pressure",
                "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "weather_code"]


def _as_float(value, field, source):
    if value is None:
        raise HTTPException(
            status_code=502,
            detail=f"Live weather feed returned no value for '{field}' ({source})",
        )
    try:
        return float(value)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=502,
            detail=f"Live weather feed returned a non-numeric '{field}' ({source})",
        )


def _daily_value(daily, key, index):
    values = (daily or {}).get(key) or []
    if not values:
        return None
    try:
        return values[index]
    except IndexError:
        return values[-1]


def _today_index(daily):
    """Index of today in a past_days+N response, falling back to the last entry."""
    times = (daily or {}).get("time") or []
    today = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    ).date().isoformat()
    if today in times:
        return times.index(today)
    return max(len(times) - 1, 0)


@app.get("/forecast")
def forecast(city: str):
    """Live prediction for a city using Open-Meteo's FREE forecast API (no key).

    Uses the SAME daily aggregates the model was trained on
    (temperature_2m_mean/max/min, wind_speed_10m_max) instead of repeating the
    current temperature into all three slots, which was a train/serve mismatch.
    """
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=15,
        )
        geo.raise_for_status()
        results = geo.json().get("results") or []
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="Geocoding service unavailable")
    if not results:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found")
    loc = results[0]

    try:
        met = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "current": ",".join(CURRENT_VARS),
                "daily": ",".join(DAILY_VARS),
                "hourly": "wind_speed_10m",
                "past_days": 1,
                "forecast_days": 7,
                "timezone": "Asia/Kolkata",
            },
            timeout=20,
        )
        met.raise_for_status()
        d = met.json()
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="Forecast service unavailable")

    current = d.get("current") or {}
    daily = d.get("daily") or {}
    idx = _today_index(daily)

    temp_mean = _as_float(_daily_value(daily, "temperature_2m_mean", idx),
                          "temperature_2m_mean", "daily")
    temp_max = _as_float(_daily_value(daily, "temperature_2m_max", idx),
                         "temperature_2m_max", "daily")
    temp_min = _as_float(_daily_value(daily, "temperature_2m_min", idx),
                         "temperature_2m_min", "daily")
    wind_speed_max = _as_float(_daily_value(daily, "wind_speed_10m_max", idx),
                               "wind_speed_10m_max", "daily")

    rh = _as_float(current.get("relative_humidity_2m"), "relative_humidity_2m", "current")
    pressure = _as_float(current.get("surface_pressure"), "surface_pressure", "current")
    cloud = _as_float(current.get("cloud_cover"), "cloud_cover", "current")
    gust = _as_float(current.get("wind_gusts_10m"), "wind_gusts_10m", "current")
    wind_dir = _as_float(current.get("wind_direction_10m"), "wind_direction_10m", "current")

    # wind_speed_mean: true mean of today's 24 hourly values, else the current reading.
    hourly = (d.get("hourly") or {}).get("wind_speed_10m") or []
    today_hours = [v for v in hourly[idx * 24:(idx + 1) * 24] if v is not None]
    wind_speed_mean = (sum(today_hours) / len(today_hours)) if today_hours else None
    wind_speed_mean = wind_speed_mean if wind_speed_mean is not None else _as_float(
        current.get("wind_speed_10m"), "wind_speed_10m", "current")

    # rain_lag1 is yesterday's total: index before today, or the first entry.
    lag_idx = idx - 1 if idx > 0 else 0
    rain_lag1 = _daily_value(daily, "precipitation_sum", lag_idx)
    rain_lag1 = 0.0 if rain_lag1 is None else float(rain_lag1)

    today_ist = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    )

    row = {
        "temp_mean": temp_mean,
        "temp_max": temp_max,
        "temp_min": temp_min,
        "relative_humidity": rh,
        "dew_point": round(dew_point(temp_mean, rh), 2),
        "surface_pressure": pressure,
        "cloud_cover": cloud,
        "wind_speed_mean": wind_speed_mean,
        "wind_speed_max": wind_speed_max,
        "wind_gust": gust,
        "wind_direction": wind_dir,
        "rain_lag1": rain_lag1,
        "day_of_year": float(today_ist.timetuple().tm_yday),
    }
    row = encode_row(row)

    prob, std = _predict_proba(row)
    reliability = assess(row, prob, ensemble_std=std, meta=_metadata, threshold=THRESHOLD)

    return {
        "city": loc["name"],
        "location": f"{loc['name']}, {loc.get('admin1', '')} ({loc.get('country_code', '')})",
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "current_weather_code": current.get("weather_code"),
        "current_conditions": WMO_CODES.get(current.get("weather_code"), "n/a"),
        "prediction": reliability["label"],
        "probability": round(prob, 4),
        "threshold": THRESHOLD,
        "risk_level": _risk_level(prob),
        "reliability": reliability,
        "explanation": _build_explanation(row),
        "model_agreement_std": round(std, 4),
        "input_features": row,
        "heavy_rain_threshold_mm": _metrics["heavy_rain_threshold_mm"],
        # Shape the live page's precipitation chart expects (Open-Meteo style).
        "daily_forecast": {
            "time": list(daily.get("time") or []),
            "precipitation_sum": [0.0 if v is None else float(v)
                                  for v in (daily.get("precipitation_sum") or [])],
        },
        "data_source": "Open-Meteo live forecast (free, no API key): daily "
                       "aggregates for temperature/wind, current readings for "
                       "humidity, pressure, cloud and gusts",
        "disclaimer": "Prototype demo: model was trained on Konkan-coast data. "
                      "Live forecast is a demonstration, not an operational product.",
    }


# --------------------------------------------------------------------------
# Batch prediction
# --------------------------------------------------------------------------
@app.post("/predict")
def predict(payload: WeatherInput):
    row = encode_row({name: float(getattr(payload, name)) for name in MODEL_FEATURES})
    prob, std = _predict_proba(row)
    reliability = assess(row, prob, ensemble_std=std, meta=_metadata, threshold=THRESHOLD)

    return {
        "prediction": reliability["label"],
        "probability": round(prob, 4),
        "threshold": THRESHOLD,
        "risk_level": _risk_level(prob),
        "reliability": reliability,
        "explanation": _build_explanation(row),
        "model_agreement_std": round(std, 4),
        "heavy_rain_threshold_mm": _metrics["heavy_rain_threshold_mm"],
        "disclaimer": _disclaimer(),
    }


app.mount("/frontend", StaticFiles(directory=str(BASE / "frontend")), name="frontend")
