"""
Step 3: Fault model.

One shared Random Forest answers: "given how THIS sensor has behaved recently
(and whether other sensors agree), is it normal, spike, stuck, noise or drift?"

Run from the sensora/ root:
    python -m ml.train_fault_model
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from ml.common import MODEL_DIR, WARMUP
from ml.generate_data import generate
from ml.features import (SENSOR_FEATURES, compute_features, fault_long,
                         fault_labels, fault_long_labels)

CLASSES = ["none", "spike", "stuck", "noise", "drift"]


# Independent streams (different seeds = different random episodes).
# Train / validation / test never share an episode -> honest test score.
TRAIN_SEEDS, VAL_SEEDS, TEST_SEEDS = [1, 2, 3], [4], [5]


def build(seeds):
    Xs, ys = [], []
    for seed in seeds:
        df = generate(n=12000, seed=seed)
        f = compute_features(df)
        X = fault_long(f)
        y = fault_long_labels(fault_labels(df))
        X["event"] = X["t"].map(df["event"] != "none").astype(bool)
        X["stream"] = seed
        keep = X["t"] >= WARMUP
        Xs.append(X[keep]); ys.append(y[keep])
    return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)


def episode_recall(X, y, pred):
    """A fault episode counts as caught if the model flags the right fault
    type at least once during it (row-level recall penalises detection lag)."""
    run = ((y != y.shift()) | (X["sensor"] != X["sensor"].shift())
           | (X["stream"] != X["stream"].shift())).cumsum()
    d = pd.DataFrame({"run": run, "y": y.values, "ok": (pred == y.values)})
    d = d[d["y"] != "none"]
    per = d.groupby("run").agg(y=("y", "first"), caught=("ok", "any"))
    print("Episode-level recall (fault caught at least once):")
    for c, g in per.groupby("y"):
        print(f"  {c:6s} {g['caught'].sum()}/{len(g)}  ({g['caught'].mean():.0%})")


def show(name, model, X, y):
    pred = model.predict(X[SENSOR_FEATURES])
    print(f"\n=== {name} ({len(y)} rows) ===")
    print(classification_report(y, pred, labels=CLASSES, digits=3, zero_division=0))
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=CLASSES),
                      index=[f"true_{c}" for c in CLASSES],
                      columns=[f"pred_{c}" for c in CLASSES])
    print(cm.to_string())
    return pred


if __name__ == "__main__":
    X_tr, y_tr = build(TRAIN_SEEDS)
    X_va, y_va = build(VAL_SEEDS)
    X_te, y_te = build(TEST_SEEDS)
    print("rows train/val/test:", len(X_tr), len(X_va), len(X_te))
    print("train class counts:\n", y_tr.value_counts().to_string())

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced_subsample",   # faults are rare; don't let 'none' dominate
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_tr[SENSOR_FEATURES], y_tr)

    pred_va = show("VALIDATION", model, X_va, y_va)
    pred_te = show("TEST (independent stream)", model, X_te, y_te)
    episode_recall(X_te, y_te, pred_te)

    imp = pd.Series(model.feature_importances_, index=SENSOR_FEATURES).sort_values(ascending=False)
    print("\nFeature importance:\n", imp.round(3).to_string())

    # Does it avoid blaming sensors during REAL events? (false-alarm check)
    ev = X_te["event"].values & (y_te == "none").values      # real event, all sensors healthy
    print(f"\nFalse-fault rate during real events, healthy sensors (test): {(pred_te[ev] != 'none').mean():.3%}")
    print(f"False-fault rate on all normal rows (test): {(pred_te[(y_te == 'none').values] != 'none').mean():.3%}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": SENSOR_FEATURES, "classes": list(model.classes_)},
                MODEL_DIR / "fault_model.joblib")
    print("\nSaved", MODEL_DIR / "fault_model.joblib")