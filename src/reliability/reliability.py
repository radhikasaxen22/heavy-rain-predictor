"""Phase 5: Reliability / uncertainty assessment.

Combines three signals into a HIGH / MEDIUM / LOW reliability score:
  A. Prediction probability          - how close the model is to a firm 0/1
  B. Out-of-distribution check       - is the input like anything we trained on?
  C. Ensemble disagreement           - do the 3 seeds agree?

Disclaimer: this is a heuristic confidence indicator, NOT a scientifically
calibrated probability of correctness.
"""
import json
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# metadata.json is immutable at runtime; read it once per process, not per request.
_META = None


def load_metadata(path=None):
    """Load (and cache) models/metadata.json, independent of the current directory."""
    global _META
    if _META is None:
        with open(path or MODELS_DIR / "metadata.json") as fh:
            _META = json.load(fh)
    return _META


def confidence_from_prob(p, threshold=0.5):
    """Distance of p from the decision boundary, normalised to the wider half."""
    span = threshold if p < threshold else 1.0 - threshold
    conf = abs(p - threshold) / span if span else 0.0
    if conf >= 0.6:
        return "HIGH"
    if conf >= 0.3:
        return "MEDIUM"
    return "LOW"


def ood_assessment(row, meta):
    """Standardised deviation of each feature from the training distribution.

    NOTE: wind_direction and day_of_year are circular and are z-scored here as
    ordinary numbers, so a 359 deg / 1 deg boundary and a Dec-31 / Jan-1 boundary
    look like large deviations. Treating them as linear is a known limitation;
    see feature_config.circular_features for the sin/cos encoding that fixes it
    once the model is retrained.
    """
    zs = []
    for feature in meta["model_features"]:
        if feature not in row or row[feature] is None:
            continue
        mu = meta["feature_mean"][feature]
        sd = meta["feature_std"][feature]
        z = abs((row[feature] - mu) / sd) if sd else 0.0
        zs.append(z)
    if not zs:
        return 0.0, True
    mean_z = sum(zs) / len(zs)
    max_z = max(zs)
    flag = (mean_z > 1.5) or (max_z > 4.0)
    return mean_z, bool(flag)


def assess(row, prob, ensemble_std=None, meta=None, threshold=0.5):
    meta = meta if meta is not None else load_metadata()

    ood_score, ood_flag = ood_assessment(row, meta)
    confidence = confidence_from_prob(prob, threshold)
    reasons = []

    if ensemble_std is not None:
        if ensemble_std > 0.15:
            confidence = "LOW" if confidence == "LOW" else "MEDIUM"
            reasons.append(f"Model ensemble disagrees strongly (std {ensemble_std:.2f})")

    if ood_flag:
        if confidence == "HIGH":
            confidence = "MEDIUM"
        reasons.append("Input conditions are unusual compared with the training data")
        if ood_score > 2.5:
            confidence = "LOW"

    if not reasons:
        reasons.append("Input lies within the training-data range and the model is confident")

    return {
        "probability": round(float(prob), 4),
        "prediction": int(prob >= threshold),
        "label": "HEAVY RAINFALL" if prob >= threshold else "NO HEAVY RAINFALL",
        "confidence": confidence,
        "ood_score": round(float(ood_score), 3),
        "ood_flag": bool(ood_flag),
        "reliability": confidence,
        "threshold": float(threshold),
        "reasons": reasons,
        "disclaimer": "Reliability is a heuristic confidence indicator, not a "
                      "scientifically calibrated probability of correctness.",
    }
