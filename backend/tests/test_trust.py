"""
Tests for the trust tracker.

Run from the sensora/ root:
    python -m backend.tests.test_trust
"""
import pandas as pd

from backend.app.services.inference import Engine
from backend.app.services.trust import TrustTracker
from ml.common import SENSORS, WARMUP
from ml.generate_data import generate


def probs(label=None, p=0.9):
    """Fake fault-model output for one sensor."""
    d = {"none": 1.0, "spike": 0.0, "stuck": 0.0, "noise": 0.0, "drift": 0.0}
    if label:
        d["none"], d[label] = 1 - p, p
    return d


def run(tracker, sensor, label, steps, p=0.9):
    """Feed `steps` readings where `sensor` shows `label` (None = healthy); others healthy."""
    for _ in range(steps):
        tracker.update({s: probs(label if s == sensor else None, p) for s in SENSORS})
    return tracker.snapshot()["sensors"][sensor]


def unit_tests():
    print("UNIT TESTS")
    checks = []

    t = TrustTracker()
    r = run(t, "co", None, 500)
    checks.append(("healthy sensor stays at 100 / OK", r["trust"] == 100 and r["status"] == "OK"))

    t = TrustTracker()
    r = run(t, "co", "spike", 30)
    low = r["trust"]
    r = run(t, "co", None, 100)
    checks.append((f"brief spike burst: trust dips to {low}, recovers to {r['trust']} / {r['status']}",
                   low < 100 and r["status"] == "OK"))

    t = TrustTracker()
    r = run(t, "voc", "stuck", 20)
    checks.append((f"short stuck (20 steps): trust {r['trust']} -> {r['status']}", r["status"] == "MONITOR"))

    t = TrustTracker()
    r = run(t, "voc", "drift", 150)
    checks.append((f"sustained drift: trust {r['trust']} -> {r['status']}", r["status"] == "RECALIBRATE"))

    t = TrustTracker()
    r = run(t, "temp", "stuck", 100)
    checks.append((f"long stuck: trust {r['trust']} -> {r['status']}", r["status"] == "REPLACE"))

    t = TrustTracker()
    run(t, "pm25", "noise", 60)
    r = t.snapshot()["sensors"]["pm10"]
    checks.append(("a faulty sensor does not reduce its neighbours' trust", r["trust"] == 100))

    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert all(ok for _, ok in checks), "a unit test failed"


def stream_demo():
    print("\nSTREAM DEMO (independent stream, seed 5)")
    df = generate(n=12000, seed=5)
    raw = Engine().run_batch(df[SENSORS])

    tracker = TrustTracker()
    series = {s: [] for s in SENSORS}
    worst = {s: (100.0, "OK") for s in SENSORS}
    for i in range(WARMUP, len(df)):
        snap = tracker.update(raw[i]["fault_probs"])
        for s in SENSORS:
            v = snap["sensors"][s]
            series[s].append(v["trust"])
            if v["trust"] < worst[s][0]:
                worst[s] = (v["trust"], v["status"])

    d = df.iloc[WARMUP:].reset_index(drop=True)
    in_event = (d["event"] != "none").values

    print(f"{'sensor':9s} {'trust while truly':>18s} {'trust in real events':>21s}  {'lowest trust':>13s}  status at lowest")
    print(f"{'':9s} {'faulty (mean)':>18s} {'(sensor healthy)':>21s}")
    for s in SENSORS:
        tr = pd.Series(series[s])
        faulty = (d["faulty_sensor"] == s).values
        # events where this sensor had no fault for the previous 400 steps
        recent_fault = pd.Series(faulty).rolling(400, min_periods=1).max().astype(bool).values
        clean_event = in_event & ~recent_fault
        ev_mean = tr[clean_event].mean() if clean_event.any() else float("nan")
        print(f"{s:9s} {tr[faulty].mean():18.1f} {ev_mean:21.1f}  {worst[s][0]:13.1f}  {worst[s][1]}")
    print("\n(trust while truly faulty should be LOW; trust during real events should stay HIGH)")


if __name__ == "__main__":
    unit_tests()
    stream_demo()