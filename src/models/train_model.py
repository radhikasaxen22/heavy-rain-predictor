"""Phase 3: Train the XGBoost heavy-rainfall classifier (small ensemble).

Train -> val -> test via a time-aware split. The served probability is the mean
of 3 seeded members; their spread feeds the reliability module.

Honest evaluation:
  * ROC-AUC and PR-AUC are computed on PROBABILITIES (the old code passed hard
    0/1 predictions to roc_auc_score, which silently produced balanced accuracy).
  * Probabilities are isotonic-calibrated on validation; the calibration knots
    are saved so the API applies exactly the same mapping at serve time.
  * The decision threshold is chosen on validation (max CSI), not assumed 0.5.

Outputs in models/:
  xgb_ens_{0,1,2}.json  - trained XGBoost models (xgboost native format)
  metrics.json          - thresholded + probabilistic metrics, per split
  metadata.json         - feature order, training mean/std (for OOD checks)
  calibration.json      - isotonic knots (x, y) for probability calibration

Run from the repository root:
    python src/models/train_model.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             roc_auc_score)

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
from features.feature_config import (  # noqa: E402
    CIRCULAR_ENCODING, CIRCULAR_FEATURES, TARGET, THRESHOLD_MM, active_features,
)

SPLIT_DIR = BASE / "data" / "processed" / "split"
MODEL_DIR = BASE / "models"
SEEDS = [42, 7, 2020]
# use_label_encoder=False was removed from XGBoost's API; it is a no-op there.
PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
              verbosity=0, n_jobs=-1)


def expand(df, names):
    """Select the active feature matrix, adding sin/cos columns when enabled."""
    if not CIRCULAR_ENCODING:
        return df[names]
    out = df.copy()
    for col, period in CIRCULAR_FEATURES.items():
        radians = 2.0 * np.pi * out[col] / period
        out[f"{col}_sin"] = np.sin(radians)
        out[f"{col}_cos"] = np.cos(radians)
    return out[names]


def threshold_metrics(y, p, threshold):
    """Classification metrics at a threshold, plus meteorology POD/FAR/CSI."""
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    far = fp / (tp + fp) if tp + fp else 0.0
    csi = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": (rec + spec) / 2.0,
        "specificity": spec,
        "precision": prec,
        "recall": rec,
        "f1": f1_score(y, pred),
        "pod": rec,
        "far": far,
        "csi": csi,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
        "n": int(len(y)),
        "n_heavy": int(y.sum()),
    }


def probabilistic_metrics(y, p):
    """Ranking and calibration quality. These need probabilities, not labels."""
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def choose_threshold(y, p):
    """Pick the validation threshold that maximises CSI (critical success index)."""
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    best_thr, best_csi = 0.5, -1.0
    for thr in grid:
        pred = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        denom = tp + fp + fn
        csi = tp / denom if denom else 0.0
        if csi > best_csi:
            best_thr, best_csi = float(thr), csi
    return best_thr


def fit_calibrator(y, p):
    """Isotonic regression on validation probabilities, clipped outside the data."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    x = getattr(iso, "X_thresholds_", None)
    yk = getattr(iso, "y_thresholds_", None)
    if x is None or yk is None or len(x) < 2:
        # Older scikit-learn: sample the fitted step function on a fixed grid.
        grid = np.linspace(0.0, 1.0, 101)
        return grid, np.clip(iso.predict(grid), 0.0, 1.0)
    x = np.asarray(x, dtype=float)
    yk = np.asarray(yk, dtype=float)
    keep = np.concatenate(([True], np.diff(x) > 0))
    return x[keep], yk[keep]


def apply_calibrator(p, x, y):
    return np.clip(np.interp(p, x, y), 0.0, 1.0)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    val = pd.read_csv(SPLIT_DIR / "val.csv")
    test = pd.read_csv(SPLIT_DIR / "test.csv")

    names = active_features()
    X_tr, y_tr = expand(train, names), train[TARGET].values
    X_val, y_val = expand(val, names), val[TARGET].values
    X_te, y_te = expand(test, names), test[TARGET].values
    print(f"[features] {len(names)} columns | circular encoding {CIRCULAR_ENCODING}")

    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    scale_pos_weight = neg / max(pos, 1)
    print(f"[train] {len(X_tr):,} rows | positives {pos} | scale_pos_weight {scale_pos_weight:.2f}")

    models = []
    for i, seed in enumerate(SEEDS):
        model = xgb.XGBClassifier(**PARAMS, random_state=seed,
                                  scale_pos_weight=scale_pos_weight)
        model.fit(X_tr, y_tr, verbose=False)
        path = MODEL_DIR / f"xgb_ens_{i}.json"
        model.save_model(str(path))
        models.append(model)
        print(f"[train] seed {seed} done -> {path}")

    def ens_prob(X):
        return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)

    raw_val, raw_test = ens_prob(X_val), ens_prob(X_te)

    cal_x, cal_y = fit_calibrator(y_val, raw_val)
    probs_val = apply_calibrator(raw_val, cal_x, cal_y)
    probs_test = apply_calibrator(raw_test, cal_x, cal_y)
    (MODEL_DIR / "calibration.json").write_text(
        json.dumps({"x": cal_x.tolist(), "y": cal_y.tolist()}, indent=2)
    )

    threshold = choose_threshold(y_val, probs_val)
    val_m = threshold_metrics(y_val, probs_val, threshold)
    test_m = threshold_metrics(y_te, probs_test, threshold)
    val_m.update(probabilistic_metrics(y_val, probs_val))
    test_m.update(probabilistic_metrics(y_te, probs_test))

    metrics = {
        "threshold_prediction": threshold,
        "threshold_source": "chosen on validation by max CSI, applied to calibrated probabilities",
        "calibrated": True,
        "heavy_rain_threshold_mm": THRESHOLD_MM,
        "val": val_m,
        "test": test_m,
        "metrics_note": (
            "roc_auc/pr_auc are computed on probabilities; brier is on calibrated "
            "probabilities. balanced_accuracy = (recall + specificity) / 2."
        ),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    metadata = {
        "model_features": names,
        "seeds": SEEDS,
        "scale_pos_weight": float(scale_pos_weight),
        "n_train": int(len(X_tr)),
        "circular_encoding": bool(CIRCULAR_ENCODING),
        "feature_mean": {c: float(X_tr[c].mean()) for c in names},
        "feature_std": {c: float(X_tr[c].std()) for c in names},
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for split, m in (("VAL", val_m), ("TEST", test_m)):
        print(f"\n== {split} ==")
        for k, v in m.items():
            if not isinstance(v, dict):
                print(f"  {k:<18} {v:.4f}" if isinstance(v, float) else f"  {k:<18} {v}")
    print(f"\n[threshold] {threshold}")
    print("[saved] models + metrics.json + metadata.json + calibration.json")


if __name__ == "__main__":
    main()
