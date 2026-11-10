# 🌧️ XAI-based Heavy Rainfall Predictor

A beginner-friendly prototype that predicts the probability of heavy rainfall
from atmospheric data, explains **why** with SHAP, scores prediction reliability,
and shows everything on a web dashboard.

> **Prototype built using publicly available atmospheric/rainfall data, with an
> INSAT-compatible pipeline designed for future integration.**

---

## Problem statement
Many lives and livelihoods depend on warnings of very heavy rainfall. ISRO
wants a system that predicts heavy rain and — just as importantly — **explains
its reasoning** and flags when the prediction may be unreliable.

## Project objective
Complete pipeline: **DATA → PREPROCESSING → FEATURES → ML MODEL → PREDICTION →
XAI → RELIABILITY → API → WEB DASHBOARD**

## System architecture
```
IMD 0.25° daily rainfall (target)   ERA5 atmosphere, daily aggregates (features)
          │                                      │
          └──────────────┬───────────────────────┘
                 data/processed/rainfall_dataset.csv
                 time-aware train/val/test split (2011–2020, Konkan coast)
                 XGBoost ensemble (3 seeds) -> SHAP -> reliability
                          │
                   FastAPI  (/predict /metrics /features)
                          │
            Static dashboard (HTML/CSS/JS, no build step)
```

## Dataset
| Role | Source | Detail | Cost |
|---|---|---|---|
| **Target** | IMD 0.25° daily gridded rainfall (Pai et al. 2014) | Gauge-based, India, 2001–2020 on disk; merged window 2011–2020 | Free |
| **Features** | ERA5 reanalysis, 0.25°, daily aggregations | Via Open-Meteo archive API (keyless) or Copernicus CDS | Free |

Region: **Konkan coast**, 15.5–20.0 °N × 72.5–74.5 °E (Mumbai→Goa), so heavy-rain
events are common enough to learn from. That box is 171 grid points at 0.25°, of
which **131 have usable IMD coverage** (≥90% of days non-null) and 40 are sea or
sparse. The rebuild samples **36 cells** (every second cell in both directions)
over **2011-01-01 → 2020-12-31**: the Open-Meteo free tier refuses a larger
request outright and rate-limits long runs, so the full 131-cell × 20-year pull is
not reachable without a paid key. Sampling on a regular stride keeps spatial
coverage of the strip instead of clustering on one end.

Row count and heavy-rain prevalence are **measured**, not asserted here:
`build_dataset.py` prints them and `train_model.py` records the window, cell count
and row count in `models/metadata.json`. The same file drives the API's
data-window disclaimer, so the served claim cannot drift from the trained facts.

Origin scripts: `src/data/download_imd_rainfall.py`,
`download_era5_features.py` (Open-Meteo, default), `download_era5_cds.py`
(same features straight from CDS; needs a free key, no quota — see its docstring),
`build_dataset.py`. All are resumable and skip work already on disk.

## Target definition
`1` if IMD daily rainfall **> 64.5 mm** else `0`. 64.5 mm is IMD's own "Very
Heavy Rain" lower category boundary. Kept as a config constant
(`THRESHOLD_MM`) so multi-class categories can be added later.

## Features (registry: `src/features/feature_config.py`)
13 model features, each with units + physical reason: `temp_mean/max/min`,
`relative_humidity`, `dew_point`, `surface_pressure`, `cloud_cover`,
`wind_speed_mean/max`, `wind_gust`, `wind_direction`, `rain_lag1` (previous day's
rain — the one truly *predictive* feature), `day_of_year` (monsoon seasonality).
Lat/lon are kept for spatial reference but excluded from the model so SHAP
explanations stay physical. Same-day features make the MVP a **diagnostic /
nowcast** rather than a true 24 h forecast — this is documented in Limitations.

## Model
**XGBoost** binary classifier (CPU-friendly, SHAP-compatible). Class imbalance
handled with `scale_pos_weight` set from the observed prevalence at training time
(≈50:1 on this window; the exact value is in `models/metadata.json`). An
**ensemble of 3 seeds** gives mean probability + disagreement signal. Time-aware
split: train 2011–2016 · val 2017–2018 · test 2019–2020 (no future leakage).

## Evaluation (test set)
| Metric | Value | Meaning |
|---|---|---|
| Balanced accuracy | 0.923 | Mean of recall and specificity; the honest headline number |
| Recall / POD | 0.915 | Catches 91.5% of real heavy-rain days |
| Precision | 0.359 | 36% of "heavy" warnings are true |
| F1 / CSI | 0.516 / 0.347 | Combined skill score |
| FAR | 0.641 | False alarms are high (rare events) |
| ROC-AUC / PR-AUC / Brier | pending | Not yet measured — see note below |

**Correction (read this before quoting a number).** Earlier versions of this
README listed "ROC-AUC 0.923". That value was **balanced accuracy mislabelled**:
the training script passed hard 0/1 predictions to `roc_auc_score`, which then
computed the mean of recall and specificity. The field is now `balanced_accuracy`
in `models/metrics.json`; `roc_auc`, `pr_auc` and `brier` are explicitly `null`
rather than guessed. `src/models/train_model.py` has been fixed to score
probabilities, so a retrain fills in the real values.

The table above describes the artifacts currently in `models/`, which were trained
before the reduced-scope rebuild and will be replaced when it lands. Their own row
counts also contradict the "41 grid cells" this README used to claim: validation
covers 34,310 rows over two years, i.e. ≈47 cells × 365 days. The old cell count
was wrong; the row count was right.

Heavy rain is rare (~2%), so high recall comes with false alarms — exactly why
the reliability module exists. `models/metrics.json` also stores the threshold
used; the shipped artifacts used a fixed 0.5, while a retrain picks the
validation threshold that maximises CSI.

## XAI (SHAP)
- **Global:** `models/global_shap.json` — mean |SHAP| per feature. Top drivers:
  `rain_lag1`, `relative_humidity`, `cloud_cover`, `dew_point` (moisture + persistence).
- **Local:** POST `/predict` returns each feature's SHAP value and impact
  ("*relative_humidity pushed towards heavy rain*"). Values are real model
  attributions — never invented text.

## Reliability methodology
Three signals combined into **HIGH / MEDIUM / LOW**:
1. **Probability** → confidence = distance from 0.5.
2. **Out-of-distribution (OOD)** check → standardised distance of each input
   feature from training mean/std.
3. **Ensemble disagreement** → std of the 3 model probabilities.

OOD or strong disagreement downgrades reliability and adds a human-readable
reason. This is a *heuristic confidence indicator*, **not** a calibrated
probability of correctness.

## API (FastAPI)
```bash
python -m uvicorn api.app:app --port 8000
```
| Endpoint | Purpose |
|---|---|
| `POST /predict` | weather JSON → prediction, probability, risk, reliability, SHAP explanation |
| `GET /forecast?city=…` | live prediction for any city via Open-Meteo (no key) |
| `GET /metrics` | model performance + global SHAP importance |
| `GET /features` | feature definitions |
| `GET /regions` | region/city registry |
| `GET /` | web dashboard |

`/predict` validates strictly: unknown fields, missing fields, strings, booleans
and out-of-range values (e.g. humidity > 100) are rejected with HTTP 422 rather
than silently coerced into a forecast.

### Live forecast — free Open-Meteo API (no key)
`GET /forecast?city=Bhubaneswar` geocodes the city (free geocoding API), pulls
**real-time** weather (free forecast API), maps it to the model's 13 features
(dew point computed by Magnus formula), and returns prediction + reliability +
SHAP. Open the dashboard → type any city → "Go". This makes the demo live with
real weather, still fully free.

The mapping uses the **same daily aggregates the model was trained on**
(`temperature_2m_mean/max/min` from the daily block, `wind_speed_10m_max`,
yesterday's `precipitation_sum` as `rain_lag1`, and the mean of today's 24 hourly
wind speeds). An earlier version copied the current temperature into all three
temperature slots — a train/serve mismatch that this fixes. Missing upstream
values return HTTP 502 naming the field instead of crashing a `float(None)`.

### Multi-region support
`data/regions.json` defines state/coast boxes: **Maharashtra (Konkan)**,
**Odisha coast**, **Kerala coast**, plus 5 demo cities (Mumbai, Ratnagiri,
Bhubaneswar, Puri, Kochi). To build training data for another region:
```powershell
$env:REGION = "odisha_coast"
python src\data\download_era5_features.py   # resumable; add cells anytime
python src\data\build_dataset.py            # -> rainfall_dataset_odisha_coast.csv
```
The currently trained/served model is the Konkan-coast one; the same pipeline
retrains per region. Live forecast reuses it at any city as a demonstration.

## Frontend
Three static pages sharing one `app.js` (no build step, no CDN deps):
`index.html` (predict form + SHAP + reliability), `live.html` (live city
forecast), `insights.html` (performance + global SHAP + methodology).
The dashboard shows prediction card, SHAP bar chart, reliability panel, model
performance + confusion matrix, 3 demo input buttons (heavy day / normal day /
unusual input), and a clearly-labelled **INSAT-3D/3DR Integration Ready** section.

Pages served by a plain static host (Vercel) cannot run Python, so they detect
that the API is absent and fall back to **precomputed outputs** shipped as
`frontend/demo_*.json`, labelled "HOSTED DEMO MODE" on screen. No live inference
happens on the hosted site.

## Installation & run locally
```powershell
git clone https://github.com/radhikasaxen22/heavy-rain-predictor.git
cd heavy-rain-predictor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt   # full dev stack (data + model + API)
# deployment-only (Vercel uses just this): python -m pip install -r requirements.txt

# 1. build the dataset (skips files already downloaded)
python src\data\download_imd_rainfall.py
python src\data\download_era5_features.py
python src\data\build_dataset.py

# 2. split + train + XAI
python src\features\prepare_split.py
python src\models\train_model.py
python src\xai\build_shap.py

# 3. run
python -m uvicorn api.app:app --port 8000
# open http://127.0.0.1:8000
```

All scripts resolve `models/`, `data/` and `frontend/` from the repository root
rather than the current directory, so they can be invoked from anywhere.

> **The training data is not in this repository.** `data/processed/**` is
> gitignored, so the committed model artifacts run but cannot be retrained or
> re-scored without first running the three download/build scripts above. That is
> why `roc_auc`, `pr_auc` and `brier` are `null` in `models/metrics.json` until
> someone retrains.

Example API call:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" ^
  -d @data/sample/sample_inputs.json
```

## Testing
```bash
python -m pytest tests -q
```
Covers heavy/normal/unusual cases, the response contract, real SHAP output,
missing / non-numeric / boolean / out-of-range / unknown fields, the live
`/forecast` mapping (with Open-Meteo mocked), the metrics honesty guard, features,
regions and dashboard endpoints (22 tests). Tests pass from any working
directory, which is the point: the original suite only passed from the repo root.

## Limitations
- MVP is trained on **public IMD + ERA5 data**, not INSAT imagery.
- Same-day ERA5 features ⇒ diagnostic/nowcast, not an operational 24 h forecast.
- ~3 h offset exists between IMD's India-standard rain day and the ERA5 day;
  matched on calendar date.
- Precision is limited by the rarity of heavy rain; reliability flags (but
  cannot fix) uncertain cases.
- Trained on the Konkan coast; performance elsewhere would need retraining.
- `wind_direction` and `day_of_year` are angles treated as linear numbers by the
  shipped models, so 359° and 1° look maximally different. `feature_config`
  contains a sin/cos encoding behind the off-by-default `CIRCULAR_ENCODING` flag;
  enabling it changes the feature count and **requires a retrain**.
- The OOD check in `reliability.py` z-scores those same circular features
  linearly and is documented as a known limitation until the flag is flipped.

## Future INSAT-3D/3DR integration
Add satellite-derived columns (brightness temperature, cloud-top temperature,
water-vapour channels) as new entries in `feature_config.FEATURES`, retrain, and
the API/dashboard/XAI/reliability adapt automatically.

## What this prototype does NOT claim
- It is **not** a production operational ISRO forecasting system.
- It does **not** use real-time satellite data or display fake satellite data.
- Reliability is **not** a scientifically calibrated probability of correctness.
- Results are for the chosen region/period only and must be re-validated before
  any operational use.

## License / data credits
Code: open source. Data: IMD (Pai et al. 2014, cite in any output), ERA5
(Copernicus Climate Change Service) served by Open-Meteo (free, non-commercial).