"""
Runs the two trained models on raw readings and returns their outputs as
plain dicts. No decision-making here - that is fusion.py's job.

Live use:   keep the last ~400 readings in a buffer -> Engine().run(buffer)
Offline:    Engine().run_batch(dataframe) -> one result per row
"""
import joblib
import pandas as pd

from ml.common import MODEL_DIR, SENSORS
from ml.features import compute_features, fault_long, event_features


class Engine:
    def __init__(self):
        fm = joblib.load(MODEL_DIR / "fault_model.joblib")
        em = joblib.load(MODEL_DIR / "event_model.joblib")
        self.fault_model, self.fault_features = fm["model"], fm["features"]
        self.event_model, self.event_features = em["model"], em["features"]
        self.fault_classes = list(fm["classes"])
        self.event_classes = list(em["classes"])

    def run_batch(self, df: pd.DataFrame) -> list[dict]:
        """df: columns = SENSORS, ordered by time. Returns one dict per row."""
        n = len(df)
        f = compute_features(df.reset_index(drop=True))

        # fault_long() stacks sensors one after another: row = sensor_index * n + time
        pf = self.fault_model.predict_proba(fault_long(f)[self.fault_features])
        pe = self.event_model.predict_proba(event_features(f)[self.event_features])

        out = []
        for i in range(n):
            out.append({
                "fault_probs": {
                    s: {c: float(pf[k * n + i][j]) for j, c in enumerate(self.fault_classes)}
                    for k, s in enumerate(SENSORS)},
                "event_probs": {c: float(pe[i][j]) for j, c in enumerate(self.event_classes)},
                "zlevel": {s: float(f[f"{s}__zlevel"].iloc[i]) for s in SENSORS},
                "n_dev": int(f["n_dev"].iloc[i]),
            })
        return out

    def run(self, buffer: pd.DataFrame) -> dict:
        """Analyse the newest reading in a rolling buffer."""
        return self.run_last(buffer)

    def run_last(self, buffer: pd.DataFrame) -> dict:
        """Fast path for live use: features come from the whole buffer
        (they need the history) but the models predict ONLY the newest row."""
        f = compute_features(buffer[SENSORS].reset_index(drop=True))
        last = f.iloc[[-1]]
        pf = self.fault_model.predict_proba(fault_long(last)[self.fault_features])  # 7 rows, SENSORS order
        pe = self.event_model.predict_proba(event_features(last)[self.event_features])[0]
        return {
            "fault_probs": {
                s: {c: float(pf[k][j]) for j, c in enumerate(self.fault_classes)}
                for k, s in enumerate(SENSORS)},
            "event_probs": {c: float(pe[j]) for j, c in enumerate(self.event_classes)},
            "zlevel": {s: float(last[f"{s}__zlevel"].iloc[0]) for s in SENSORS},
            "n_dev": int(last["n_dev"].iloc[0]),
        }