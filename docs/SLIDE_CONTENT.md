# 📊 One-page slide content (copy-paste ready)

## Slide 1 — Title
**XAI-based Heavy Rainfall Predictor**
"Predict, Explain, Trust" — prototype on public data, INSAT-ready.

## Slide 2 — Why + Idea
> Daily >64.5 mm (IMD "Very Heavy Rain") is rare but deadly.
> A useful system must say **what** (probability), **why** (SHAP), and
> **how trustworthy** (reliability). Ours does all three in one pipeline.

## Slide 3 — Data
| Role | Source | Detail |
|---|---|---|
| Target | IMD 0.25° gridded rainfall | India, gauge-based, 2001–2020 |
| Features | ERA5 via Open-Meteo | 13 daily atmospheric variables |
| Region | Konkan coast (Mumbai→Goa) | 41 cells, **343,294 rows** |
| Balance | heavy-rain days | **2.24%** (7,686 events) |

## Slide 4 — Method
- **Model:** XGBoost ensemble (3 seeds), `scale_pos_weight ~48`, CPU-only.
- **Split:** by time — train 2001–2016 · val 2017–18 · test 2019–20 (no leakage).
- **XAI:** SHAP — global importance + per-prediction attributions.
- **Reliability:** probability confidence + out-of-distribution distance + ensemble disagreement → HIGH/MEDIUM/LOW.

## Slide 5 — Results (held-out test 2019–2020)
| Metric | Value |
|---|---|
| Balanced accuracy | **0.923** |
| Recall / POD | **0.915** |
| Precision | 0.359 |
| Specificity | 0.931 |
| F1 | 0.516 |
| CSI | 0.347 |
| FAR | 0.641 |
| ROC-AUC / PR-AUC / Brier | pending retrain |

Balanced accuracy = mean of recall and specificity. An earlier version of this
slide called it "ROC-AUC"; that was a mislabel, since the training script scored
hard 0/1 predictions instead of probabilities. The metric is now named correctly,
and ROC-AUC/PR-AUC/Brier are listed as pending rather than estimated.

Key message: catches **91.5%** of real heavy-rain days; false alarms are visible
and **flagged** by the reliability module, not hidden.

## Slide 6 — Global feature importance (mean |SHAP|)
1. rain_lag1 (persistence) — 2.07
2. relative_humidity — 1.51
3. cloud_cover — 0.91
4. dew_point — 0.49
5. surface_pressure — 0.37

Meteorologically sensible: moisture + persistence dominate.

## Slide 7 — Live demo (3 clicks)
Heavy case → **77% probability, HIGH risk, "humidity pushed towards heavy rain"**
Normal day → probability drops, explanation flips blue.
Unusual input → **reliability downgraded to LOW** with reason "input is out-of-distribution".

## Slide 8 — INSAT-3D/3DR future
Satellite columns (brightness temp, cloud-top temp, WV channels) plug into the
feature registry → retrain → same API/dashboard. **No redesign.**

## Honesty line (keep on screen)
"Prototype uses public IMD/ERA5 data; reliability is a heuristic indicator; not
an operational ISRO forecasting system."