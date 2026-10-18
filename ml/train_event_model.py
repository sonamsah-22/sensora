"""
Step 4: Event model.

Looks at the WHOLE environment (all 7 sensors + how they agree) and answers:
"is a real event happening: none, smoke_fire, gas_leak or poor_air_quality?"

Important: this model is trained to be a pure event detector. Whether a
suspicious reading is a sensor fault is decided later by the fusion layer
(Step 5), which combines this model with the fault model.

Run from the sensora/ root:
    python -m ml.train_event_model
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from ml.common import MODEL_DIR, WARMUP
from ml.generate_data import generate
from ml.features import compute_features, event_features

CLASSES = ["none", "smoke_fire", "gas_leak", "poor_air_quality"]
TRAIN_SEEDS, VAL_SEEDS, TEST_SEEDS = [1, 2, 3], [4], [5]   # same streams as Step 3


def build(seeds):
    Xs, ys, meta = [], [], []
    for seed in seeds:
        df = generate(n=12000, seed=seed)
        f = compute_features(df)
        X = event_features(f)
        keep = df["t"] >= WARMUP
        Xs.append(X[keep])
        ys.append(df.loc[keep, "event"])
        meta.append(pd.DataFrame({"fault": df.loc[keep, "fault"],
                                  "stream": seed}))
    return (pd.concat(Xs, ignore_index=True),
            pd.concat(ys, ignore_index=True),
            pd.concat(meta, ignore_index=True))


def show(name, model, X, y):
    pred = model.predict(X)
    print(f"\n=== {name} ({len(y)} rows) ===")
    print(classification_report(y, pred, labels=CLASSES, digits=3, zero_division=0))
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=CLASSES),
                      index=[f"true_{c}" for c in CLASSES],
                      columns=[f"pred_{c}" for c in CLASSES])
    print(cm.to_string())
    return pred


def episode_recall(y, meta, pred):
    run = ((y != y.shift()) | (meta["stream"] != meta["stream"].shift())).cumsum()
    d = pd.DataFrame({"run": run, "y": y.values, "ok": pred == y.values})
    d = d[d["y"] != "none"]
    per = d.groupby("run").agg(y=("y", "first"), caught=("ok", "any"))
    print("\nEpisode-level recall (event caught at least once):")
    for c, g in per.groupby("y"):
        print(f"  {c:17s} {g['caught'].sum()}/{len(g)}  ({g['caught'].mean():.0%})")


if __name__ == "__main__":
    X_tr, y_tr, _ = build(TRAIN_SEEDS)
    X_va, y_va, _ = build(VAL_SEEDS)
    X_te, y_te, m_te = build(TEST_SEEDS)
    print("rows train/val/test:", len(X_tr), len(X_va), len(X_te))
    print("train class counts:\n", y_tr.value_counts().to_string())

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_tr, y_tr)

    show("VALIDATION", model, X_va, y_va)
    pred_te = show("TEST (independent stream)", model, X_te, y_te)
    episode_recall(y_te, m_te, pred_te)

    imp = pd.Series(model.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    print("\nTop 10 features:\n", imp.head(10).round(3).to_string())

    # THE key honesty check: does a single faulty sensor fool the event model?
    during_fault = (m_te["fault"] != "none").values
    fooled = (pred_te[during_fault] != "none")
    print(f"\nEvent model raises an event during a SENSOR FAULT: {fooled.mean():.2%} of fault rows")
    if fooled.any():
        print(pd.Series(pred_te[during_fault][fooled]).value_counts().to_string())
        by_fault = (pd.Series(fooled, index=m_te.index[during_fault])
                    .groupby(m_te.loc[during_fault, "fault"]).mean())
        print("by fault type:\n", by_fault.round(3).to_string())
    print("(the fusion layer in Step 5 uses the fault model to catch these)")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": list(X_tr.columns), "classes": list(model.classes_)},
                MODEL_DIR / "event_model.joblib")
    print("\nSaved", MODEL_DIR / "event_model.joblib")