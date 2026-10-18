"""
Scenario tests for the fusion layer (hand-built situations, incl. hard cases).

Run from the sensora/ root:
    python -m backend.tests.test_fusion
"""
from collections import Counter

import numpy as np
import pandas as pd

from backend.app.services.fusion import fuse
from backend.app.services.inference import Engine
from ml.common import SENSORS
from ml.generate_data import baseline, inject_event, inject_fault

N, START, RUNS = 1000, 500, 30
EVENT_OK = {"GENUINE_EVENT", "GENUINE_EVENT_DEGRADED"}


def finalize(df):
    for c in ["pm25", "pm10", "co", "voc"]:
        df[c] = df[c].clip(lower=0)
    df["humidity"] = df["humidity"].clip(0, 100)
    return df.round(3)


# Each scenario returns (dataframe, index at which we analyse)
def normal(rng, **_):
    return finalize(baseline(N, rng)), START + 40


def event(kind):
    def make(rng, **_):
        df = baseline(N, rng)
        s, e = inject_event(df, START, kind, rng)
        return finalize(df), (s + e) // 2
    return make


def fault(kind, sensor=None):
    def make(rng, **_):
        df = baseline(N, rng)
        sen = sensor or rng.choice(SENSORS)
        s, e, rows = inject_fault(df, START, kind, sen, rng)
        if kind == "spike":
            t = max(rows) + 3
        elif kind == "drift":
            t = min(e - 5, START + 100)
        else:
            t = min(e - 5, START + 25)
        return finalize(df), t
    return make


def event_plus_fault(ev_kind, f_kind, sensor):
    """Hard case: a real event while one of its sensors is broken."""
    def make(rng, **_):
        df = baseline(N, rng)
        s, e = inject_event(df, START, ev_kind, rng)
        inject_fault(df, START, f_kind, sensor, rng)
        return finalize(df), (s + e) // 2
    return make


SCENARIOS = [
    # name,                               builder,                              expected set
    ("normal",                            normal,                               {"NORMAL"}),
    ("smoke_fire",                        event("smoke_fire"),                  EVENT_OK),
    ("gas_leak",                          event("gas_leak"),                    EVENT_OK),
    ("poor_air_quality",                  event("poor_air_quality"),            EVENT_OK),
    ("spike (random sensor)",             fault("spike"),                       {"SENSOR_FAULT"}),
    ("stuck (random sensor)",             fault("stuck"),                       {"SENSOR_FAULT"}),
    ("noise (random sensor)",             fault("noise"),                       {"SENSOR_FAULT"}),
    ("drift (random sensor)",             fault("drift"),                       {"SENSOR_FAULT"}),
    ("TRAP: VOC drift looks like leak",   fault("drift", "voc"),                {"SENSOR_FAULT"}),
    ("TRAP: huge VOC spike",              fault("spike", "voc"),                {"SENSOR_FAULT"}),
    ("HARD: smoke + stuck pm25",          event_plus_fault("smoke_fire", "stuck", "pm25"), {"GENUINE_EVENT_DEGRADED", "GENUINE_EVENT"}),
    ("HARD: smoke + stuck co",            event_plus_fault("smoke_fire", "stuck", "co"),   {"GENUINE_EVENT_DEGRADED", "GENUINE_EVENT"}),
    ("HARD: gas leak + noisy co2",        event_plus_fault("gas_leak", "noise", "co2"),    EVENT_OK),
]

if __name__ == "__main__":
    engine = Engine()
    print(f"{'scenario':36s} {'pass':>6s}   decisions seen")
    print("-" * 100)
    hazards_missed = false_alarms = 0
    for name, build, expected in SCENARIOS:
        seen = Counter()
        for seed in range(RUNS):
            rng = np.random.default_rng(1000 + seed)
            df, t = build(rng)
            raw = engine.run(df.iloc[: t + 1])         # only data up to time t (as in live use)
            d = fuse(raw)
            seen[d["decision"]] += 1
            if expected & EVENT_OK and d["alarm_action"] == "SUPPRESS_ALARM":
                hazards_missed += 1
            if not (expected & EVENT_OK) and d["alarm_action"] == "TRIGGER_ALARM":
                false_alarms += 1
        ok = sum(seen[k] for k in expected)
        print(f"{name:36s} {ok:3d}/{RUNS}   {dict(seen)}")
    print("-" * 100)
    print(f"Hazards wrongly SUPPRESSED : {hazards_missed}")
    print(f"False ALARMS triggered     : {false_alarms}")