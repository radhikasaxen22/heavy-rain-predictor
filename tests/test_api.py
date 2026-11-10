"""API tests for the heavy-rainfall predictor.

Run from ANY directory (paths are resolved from this file, not the CWD):
    python -m pytest tests -q
"""
import datetime
import json
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from fastapi.testclient import TestClient  # noqa: E402

import api.app as api_module  # noqa: E402
from api.app import app  # noqa: E402
from src.features.feature_config import MODEL_FEATURES  # noqa: E402

client = TestClient(app)

with open(BASE / "data" / "sample" / "sample_inputs.json") as fh:
    CASES = json.load(fh)["cases"]

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))


def _full(payload):
    row = dict(payload)
    for f in MODEL_FEATURES:
        if f not in row:
            row[f] = 50.0
    return row


# --------------------------------------------------------------------------
# /predict
# --------------------------------------------------------------------------
def test_heavy_rain_case():
    r = client.post("/predict", json=_full(CASES["heavy_rain_day"]))
    assert r.status_code == 200
    d = r.json()
    assert 0.0 <= d["probability"] <= 1.0
    assert d["probability"] > 0.7, d
    assert d["prediction"] == "HEAVY RAINFALL"
    assert d["reliability"]["prediction"] == 1
    assert d["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_response_contract():
    d = client.post("/predict", json=_full(CASES["normal_day"])).json()
    for key in ("prediction", "probability", "threshold", "risk_level", "reliability",
                "explanation", "model_agreement_std", "heavy_rain_threshold_mm",
                "disclaimer"):
        assert key in d, f"missing response key: {key}"
    assert 0.0 < d["threshold"] < 1.0
    assert d["heavy_rain_threshold_mm"] == 64.5


def test_explanation_is_real_shap():
    d = client.post("/predict", json=_full(CASES["heavy_rain_day"])).json()
    exps = d["explanation"]
    assert exps, "explanation must not be empty"
    assert len(exps) <= 8
    assert all({"feature", "value", "shap", "direction", "impact", "unit"} <= set(e)
               for e in exps)
    # Contributions are ordered by magnitude and any single one is plausible.
    mags = [abs(e["shap"]) for e in exps]
    assert mags == sorted(mags, reverse=True)
    assert any(e["shap"] != 0 for e in exps)

    # Averaged over 3 members, the attributions must not equal one member's.
    from src.xai.explain import explain_local
    assert len(explain_local(_full(CASES["heavy_rain_day"]))) == len(MODEL_FEATURES)


def test_normal_day_case():
    d = client.post("/predict", json=_full(CASES["normal_day"])).json()
    assert d["probability"] < 0.4, d
    assert d["prediction"] == "NO HEAVY RAINFALL"


def test_unusual_input_flags_ood():
    d = client.post("/predict", json=_full(CASES["unusual_oceanic_input"])).json()
    assert d["reliability"]["ood_flag"] is True, d


def test_missing_feature_returns_422():
    bad = _full(CASES["normal_day"])
    bad.pop("cloud_cover")
    assert client.post("/predict", json=bad).status_code == 422


def test_non_numeric_returns_422():
    bad = _full(CASES["normal_day"])
    bad["cloud_cover"] = "abc"
    assert client.post("/predict", json=bad).status_code == 422


def test_boolean_is_not_a_number():
    """True used to sail through as 1.0; it is not a humidity reading."""
    bad = _full(CASES["normal_day"])
    bad["cloud_cover"] = True
    assert client.post("/predict", json=bad).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("relative_humidity", 150), ("relative_humidity", -5),
    ("cloud_cover", 101), ("day_of_year", 0), ("surface_pressure", 10),
])
def test_out_of_range_returns_422(field, value):
    bad = _full(CASES["normal_day"])
    bad[field] = value
    r = client.post("/predict", json=bad)
    assert r.status_code == 422, r.text


def test_unknown_field_returns_422():
    bad = _full(CASES["normal_day"])
    bad["rainfall_mm_typo"] = 3.0
    assert client.post("/predict", json=bad).status_code == 422


# --------------------------------------------------------------------------
# /forecast (network mocked)
# --------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _fake_open_meteo(daily_days=8):
    today = datetime.datetime.now(IST).date()
    times, tmean, tmax, tmin, wmax, precip = [], [], [], [], [], []
    for offset in range(-1, daily_days - 1):
        times.append((today + datetime.timedelta(days=offset)).isoformat())
        tmean.append(27.0)
        tmax.append(31.0)
        tmin.append(24.0)
        wmax.append(22.0)
        precip.append(30.0 if offset == -1 else 0.0)
    hourly_time, hourly_wind = [], []
    for offset in range(-1, daily_days - 1):
        for hour in range(24):
            hourly_time.append(f"{(today + datetime.timedelta(days=offset)).isoformat()}T{hour:02d}:00")
            hourly_wind.append(12.0)
    return {
        "current": {
            "relative_humidity_2m": 88, "cloud_cover": 90, "surface_pressure": 1005,
            "wind_speed_10m": 15.0, "wind_direction_10m": 210, "wind_gusts_10m": 40.0,
            "weather_code": 65,
        },
        "daily": {
            "time": times, "temperature_2m_mean": tmean, "temperature_2m_max": tmax,
            "temperature_2m_min": tmin, "wind_speed_10m_max": wmax,
            "precipitation_sum": precip,
        },
        "hourly": {"time": hourly_time, "wind_speed_10m": hourly_wind},
    }


def _install_fake(monkeypatch, met_payload=None, geo_payload=None):
    met_payload = met_payload if met_payload is not None else _fake_open_meteo()
    geo_payload = geo_payload if geo_payload is not None else {
        "results": [{"name": "Ratnagiri", "latitude": 17.0, "longitude": 73.3,
                     "admin1": "Maharashtra", "country_code": "IN"}]
    }

    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(geo_payload if "geocoding" in url else met_payload)

    monkeypatch.setattr(api_module.requests, "get", fake_get)


def test_forecast_uses_daily_aggregates(monkeypatch):
    """Regression: /forecast used to copy one current temperature into
    temp_mean/temp_max/temp_min, a train/serve mismatch."""
    _install_fake(monkeypatch)
    r = client.get("/forecast", params={"city": "Ratnagiri"})
    assert r.status_code == 200, r.text
    d = r.json()

    feats = d["input_features"]
    assert feats["temp_mean"] == 27.0
    assert feats["temp_max"] == 31.0
    assert feats["temp_min"] == 24.0
    assert feats["wind_speed_max"] == 22.0
    assert feats["wind_speed_mean"] == pytest.approx(12.0)   # hourly mean, not current
    assert feats["rain_lag1"] == 30.0                        # yesterday, not today
    assert feats["day_of_year"] == float(datetime.datetime.now(IST).timetuple().tm_yday)
    assert 0.0 <= d["probability"] <= 1.0
    assert d["location"].startswith("Ratnagiri")
    assert d["current_conditions"] == "Heavy rain"


def test_forecast_returns_chart_series(monkeypatch):
    _install_fake(monkeypatch)
    d = client.get("/forecast", params={"city": "Ratnagiri"}).json()
    series = d["daily_forecast"]
    assert len(series["time"]) == len(series["precipitation_sum"]) >= 7
    assert all(isinstance(v, float) for v in series["precipitation_sum"])
    assert d["heavy_rain_threshold_mm"] == 64.5


def test_forecast_handles_null_daily_values(monkeypatch):
    payload = _fake_open_meteo()
    payload["daily"]["temperature_2m_mean"][1] = None
    _install_fake(monkeypatch, met_payload=payload)
    r = client.get("/forecast", params={"city": "Ratnagiri"})
    assert r.status_code == 502
    assert "temperature_2m_mean" in r.json()["detail"]


def test_forecast_unknown_city_is_404(monkeypatch):
    _install_fake(monkeypatch, geo_payload={"results": []})
    assert client.get("/forecast", params={"city": "Nowhere"}).status_code == 404


# --------------------------------------------------------------------------
# Metadata endpoints
# --------------------------------------------------------------------------
def test_metrics_endpoint():
    d = client.get("/metrics").json()
    assert "metrics" in d and "global_shap" in d
    test = d["metrics"]["test"]
    assert 0.5 < test["balanced_accuracy"] <= 1.0
    # Honesty guard: the old roc_auc field held balanced accuracy and was
    # withdrawn. It must stay null until a retrain measures it for real.
    assert test["roc_auc"] is None
    assert test["pr_auc"] is None
    assert test["brier"] is None


def test_features_endpoint():
    d = client.get("/features").json()
    assert len(d["model_features"]) == len(MODEL_FEATURES)
    assert all({"name", "unit", "why"} <= set(f) for f in d["features"])


def test_regions_endpoint():
    d = client.get("/regions").json()
    assert "maharashtra_konkan" in d["regions"] and "odisha_coast" in d["regions"]
    assert "cities" in d and len(d["cities"]) >= 5


def test_dashboard_overview():
    r = client.get("/")
    assert r.status_code == 200
    assert "Heavy Rainfall Predictor" in r.text
