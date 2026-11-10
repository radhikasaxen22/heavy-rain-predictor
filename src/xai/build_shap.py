"""Phase 4: Global SHAP feature importance.

Loads a sample of the training split, computes SHAP values for EVERY ensemble
member, and saves the mean absolute attribution across members to
models/global_shap.json. Averaging matters: the API predicts with the ensemble,
so importance from member 0 alone misrepresented what the service does.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
from features.feature_config import TARGET, active_features  # noqa: E402
from models.train_model import SEEDS, expand  # noqa: E402

SPLIT_DIR = BASE / "data" / "processed" / "split"
SAMPLE = 20000


def main():
    names = active_features()
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    pos = train[train[TARGET] == 1]
    neg = train[train[TARGET] == 0].sample(
        n=min(len(pos) * 5, SAMPLE - len(pos)), random_state=1)
    sample = pd.concat([pos, neg]).sample(frac=1, random_state=1).reset_index(drop=True)
    X = expand(sample, names)
    print(f"[sample] {len(X)} rows ({int(sample[TARGET].sum())} heavy)")

    per_member = []
    base_values = []
    for i, seed in enumerate(SEEDS):
        model = xgb.XGBClassifier()
        model.load_model(str(BASE / "models" / f"xgb_ens_{i}.json"))
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X, check_additivity=False)
        per_member.append(values)
        base_values.append(float(np.ravel(explainer.expected_value)[0]))
        print(f"[shap] member {i} (seed {seed}) done")

    shap_values = np.mean(per_member, axis=0)
    imp = np.abs(shap_values).mean(axis=0)
    rows = [{"feature": f, "importance": float(v)} for f, v in zip(names, imp)]
    rows.sort(key=lambda r: r["importance"], reverse=True)

    out = BASE / "models" / "global_shap.json"
    out.write_text(json.dumps({
        "base_value": float(np.mean(base_values)),
        "members_averaged": len(SEEDS),
        "importance": rows,
    }, indent=2) + "\n")

    print("== global feature importance (mean |SHAP| across ensemble) ==")
    for r in rows:
        print(f"  {r['feature']:<22} {r['importance']:.4f}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
