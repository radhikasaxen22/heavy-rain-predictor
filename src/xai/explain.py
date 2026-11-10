"""Phase 4 (local): explain one individual prediction with SHAP.

Uses XGBoost's native TreeSHAP (pred_contribs) so the server needs NO shap
package and NO pandas - keeping the deployment small. The SHAP values are the
same ones the shap package would produce.

Attributions are averaged across ALL ensemble members, matching what the API
actually predicts (the mean of the 3 probabilities). Averaging log-odds
contributions is not identical to the SHAP of the averaged probability - the
probabilities are a sigmoid of the margins - so the values explain the
ensemble consensus, not a single member and not the exact averaging step.
"""
from pathlib import Path

import numpy as np
import xgboost as xgb

from features.feature_config import MODEL_FEATURES

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
N_MEMBERS = 3
_STATE = {}


def get_models():
    """Load every ensemble member once per process, independent of the CWD."""
    if "models" not in _STATE:
        boosters = []
        for i in range(N_MEMBERS):
            booster = xgb.Booster()
            booster.load_model(str(MODELS_DIR / f"xgb_ens_{i}.json"))
            boosters.append(booster)
        _STATE["models"] = boosters
    return _STATE["models"]


def explain_local(row):
    """Return per-feature SHAP attributions averaged over the ensemble."""
    boosters = get_models()
    X = np.array([[float(row[f]) for f in MODEL_FEATURES]])
    dmat = xgb.DMatrix(X, feature_names=MODEL_FEATURES)

    contribs = np.mean(
        [b.predict(dmat, pred_contribs=True)[0] for b in boosters], axis=0
    )

    items = []
    for feature, value, sh in zip(MODEL_FEATURES, X[0], contribs[:-1]):
        items.append({
            "feature": feature,
            "value": round(float(value), 2),
            "shap": round(float(sh), 4),
            "direction": "increased" if sh >= 0 else "decreased",
            "impact": "pushed towards heavy rain" if sh >= 0 else "pushed away from heavy rain",
        })
    items.sort(key=lambda i: abs(i["shap"]), reverse=True)
    return items
