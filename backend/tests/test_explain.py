"""
Prints the full explanation for hand-built scenarios so you can read them.

Run from the sensora/ root:
    python -m backend.tests.test_explain
"""
import numpy as np

from backend.app.services.explain import explain
from backend.app.services.fusion import fuse
from backend.app.services.inference import Engine
from backend.app.services.trust import TrustTracker
from backend.tests.test_fusion import (normal, event, fault, event_plus_fault)

SHOW = [
    ("Normal operation",                   normal),
    ("Real smoke event",                   event("smoke_fire")),
    ("Single sensor spike (VOC)",          fault("spike", "voc")),
    ("Frozen temperature sensor",          fault("stuck", "temp")),
    ("Noisy humidity sensor",              fault("noise", "humidity")),
    ("Drifting VOC (looks like a leak)",   fault("drift", "voc")),
    ("HARD: smoke + frozen PM2.5",         event_plus_fault("smoke_fire", "stuck", "pm25")),
]


def print_result(title, res, fused):
    print("=" * 90)
    print(f"{title.upper()}\n  -> {fused['decision']}  |  alarm: {fused['alarm_action']}")
    print(f"\n{res['headline']}  [{res['severity']}]")
    print(f"{res['summary']}")
    if res["evidence"]:
        print("\nEvidence:")
        for e in res["evidence"]:
            tag = {"fault": "FAULT", "possible_fault": "POSSIBLE FAULT"}.get(e["kind"], "EVENT")
            print(f"  [{tag}] {e['text']}")
    print("\nWhy:")
    for w in res["why"]:
        print(f"  - {w}")
    print("\nActions:")
    for a in res["recommended_actions"]:
        print(f"  > {a}")
    print()


if __name__ == "__main__":
    engine = Engine()
    for title, build in SHOW:
        rng = np.random.default_rng(1003)
        df, t = build(rng)
        buf = df.iloc[: t + 1]

        raw_all = engine.run_batch(buf)                 # all rows, so trust can build up
        tracker = TrustTracker()
        for r in raw_all[-80:]:
            tracker.update(r["fault_probs"])

        raw = raw_all[-1]
        fused = fuse(raw)
        print_result(title, explain(fused, raw, buf, tracker.snapshot()), fused)