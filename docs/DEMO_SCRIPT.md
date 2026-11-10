# 🎤 Demo Script (5-minute live walkthrough)

Setup: server already running, dashboard open in fullscreen, desktop clean.

---

## 0:00 — Intro (30 s)
> "We built an XAI-based heavy rainfall predictor. It answers three questions:
> **What** is the risk? **Why** does the model think so? **How much** should we
> trust it?"

## 0:30 — The problem (30 s)
> "Very heavy rain (>64.5 mm/day) is rare but dangerous. ISRO needs predictions
> that also **explain themselves** and flag unreliable cases. Our pipeline:
> data → features → model → prediction → XAI → reliability → dashboard."

## 1:00 — Data (30 s)
> "Target: IMD's official 0.25° daily gridded rainfall (2001–2020, Konkan coast).
> Features: ERA5 atmosphere (temperature, humidity, pressure, wind, clouds)
> served free by Open-Meteo. 343,000+ daily rows, 41 coastal cells. Heavy-rain
> days are only 2.2% of the data — realistically imbalanced."

## 1:30 — Model (30 s)
> "XGBoost ensemble of 3 seeds — fast, CPU-only, SHAP-compatible. Time-aware
> split (train 2001–2016, val 2017–18, test 2019–20) so the model never sees
> the future. Test results: **balanced accuracy 0.92, catches 91.5% of real
> heavy-rain days.** Precision is lower (~36%) because heavy rain is rare —
> that's why reliability matters."

## 2:00 — LIVE: Heavy-rain case (45 s)
> 1. Click **"Heavy rain case"**.
> 2. Point at the card: *"76.7% probability, HIGH risk."*
> 3. Point at the SHAP chart: *"The model says relative humidity and cloud cover
>    pushed the risk up — the moisture story is exactly what a meteorologist
>    would say."* (red = pushed towards heavy rain)

## 2:45 — LIVE: Normal day (30 s)
> 1. Click **"Normal day"**.
> 2. *"Probability drops; the explanation flips to blue — dry conditions pushed
>    away from heavy rain."*

## 3:15 — Reliability (45 s)
> 1. Click **"Unusual input"** (we injected 850 hPa pressure, 5% humidity).
> 2. *"Reliability is now downgraded to LOW and the reason is shown: the input
>    is out-of-distribution — unlike anything we trained on. The system says
>    'don't trust this' instead of silently giving a number."*
> 3. Show the reliability method line: probability + OOD check + ensemble agreement.

## 4:00 — Model performance (30 s)
> *"Table: balanced accuracy, precision, recall, F1, specificity, plus
> meteorology scores POD/FAR/CSI and the confusion matrix on the held-out test
> set. ROC-AUC and PR-AUC are shown as pending - we do not quote a number we
> have not measured."*

## 4:30 — INSAT future + close (30 s)
> *"Everything is INSAT-ready: satellite features (brightness temperature,
> cloud-top temperature, water-vapour channels) slot in as new columns in the
> feature registry — no redesign."*
> *"And to be honest: this is a prototype on public data, not an operational ISRO
> system — but the full pipeline, explanations and reliability logic are real."*

---

## Judge Q&A — quick answers
- **Why XGBoost?** Fast, CPU-only, explainable, great with tabular weather data.
- **Why not deep learning?** Rare-event + tabular + explainability → XGBoost wins for an MVP.
- **Is this a forecast?** It's a same-day diagnostic; a 1-day-ahead variant = shift features back 1 day (documented in README).
- **Why 64.5 mm?** IMD's own "Very Heavy Rain" category boundary.
- **Data leakage?** Time-first split + only previous-day rainfall as lag — no future info.
- **What's the accuracy?** Balanced accuracy 92.3% (recall 91.5%, specificity 93.1%). Plain accuracy is 93% but that is flattered by the 98% majority class, so we lead with balanced accuracy and POD/CSI.
- **How would INSAT help?** Real-time satellite features → faster, wider, satellite-based inputs to the same pipeline.
