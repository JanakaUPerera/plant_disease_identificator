"""
Classical baseline (Question 3.a): scaled hand-crafted features + a classical classifier.

Predictions use the same crop rule as the ANN: scores of classes that do not belong to the image's
crop are removed before the highest-scoring class is chosen. Every configuration is trained on train,
scored on val_lab and val_wild with training.scores(), and logged to results/baseline_summary.csv.
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from features import GROUPS
from training import scores

FEATURE_SETS = {"colour": [GROUPS["colour"]], "texture": [GROUPS["lbp"], GROUPS["glcm"]],
                "all": [GROUPS["colour"], GROUPS["lbp"], GROUPS["glcm"]]}


def select(X, feature_set):
    """The columns of one feature set."""
    return np.concatenate([X[:, s] for s in FEATURE_SETS[feature_set]], axis=1)


def make_model(kind, seed=42, **params):
    """A scaler + classifier pipeline. Scaling gives every feature mean 0 and spread 1, which linear
    models and SVMs need; trees do not, but it does them no harm."""
    if kind == "logreg":
        clf = LogisticRegression(max_iter=3000, random_state=seed, **params)
    elif kind == "linear_svm":
        clf = LinearSVC(max_iter=20000, random_state=seed, **params)
    elif kind == "rbf_svm":
        clf = SVC(kernel="rbf", gamma="scale", random_state=seed, **params)
    elif kind == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed, **params)
    elif kind == "grad_boost":
        clf = HistGradientBoostingClassifier(max_iter=300, random_state=seed, **params)
    else:
        raise ValueError(f"unknown model kind {kind}")
    return make_pipeline(StandardScaler(), clf)


def masked_predict(model, X, crop_idx, crop_mask):
    """Class numbers, choosing only among the classes of each image's crop."""
    s = model.predict_proba(X) if hasattr(model, "predict_proba") else model.decision_function(X)
    full = np.full((len(X), crop_mask.shape[1]), -np.inf)
    full[:, model.classes_] = s
    full[~crop_mask[crop_idx]] = -np.inf
    return full.argmax(axis=1)


def run_config(config_id, kind, feature_set, params, data, class_ids, crop_mask,
               results_dir="results", model_dir="checkpoints"):
    """Train one configuration, score it on both validation sets, and log one summary row."""
    out = Path(results_dir) / "baseline_summary.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(old) and config_id in set(old["config_id"]):
        raise FileExistsError(f"{config_id} is already in {out}: use a new config_id")
    Xtr, ytr, _ = data["train"]
    model = make_model(kind, **params)
    t0 = time.perf_counter()
    model.fit(select(Xtr, feature_set), ytr)
    fit_seconds = time.perf_counter() - t0
    row = {"config_id": config_id, "kind": kind, "feature_set": feature_set, "params": str(params),
           "fit_minutes": fit_seconds / 60}
    for split in ("val_lab", "val_wild"):
        X, y, crop = data[split]
        t0 = time.perf_counter()
        pred = masked_predict(model, select(X, feature_set), crop, crop_mask)
        ms = 1000 * (time.perf_counter() - t0) / len(y)
        for k, v in scores(y, pred, class_ids).items():
            row[f"{split}_{k}"] = v
        row[f"{split}_ms_per_image"] = ms
    Path(model_dir).mkdir(exist_ok=True)
    joblib.dump(model, Path(model_dir) / f"{config_id}.joblib")
    out.parent.mkdir(exist_ok=True)
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(out, index=False)
    print(f"{config_id}: {kind:13s} {feature_set:7s} {str(params):38s} | fit {fit_seconds:6.1f} s | "
          f"val_lab F1-16 {row['val_lab_macro_f1_16_disease']:.3f} | "
          f"val_wild F1-16 {row['val_wild_macro_f1_16_disease']:.3f}")
    return row