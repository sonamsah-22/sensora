"""
Scenario simulator: produces a realistic stream the dashboard can play back.

A timeline = PREFILL healthy readings (loaded into the buffer up front) followed by
readings in which the chosen situation unfolds. Ground-truth labels are kept for
every row so the demo can show "what was really happening".
"""
import numpy as np
import pandas as pd

from backend.app.config import PREFILL, SIM_STEPS_AFTER_PREFILL
from ml.common import SENSORS
from ml.generate_data import baseline, inject_event, inject_fault

SCENARIOS = {
    "normal":           ("Normal fluctuations", "Healthy sensors, no incident.", False),
    "smoke_fire":       ("Smoke / fire event", "Several sensors rise together (PM, CO, VOC, temperature).", False),
    "gas_leak":         ("Gas leak", "VOC surges with a smaller CO and CO2 rise.", False),
    "poor_air_quality": ("Poor air quality", "Particulates and CO2 rise moderately.", False),
    "spike":            ("Sensor spike", "One sensor jumps suddenly; nothing else changes.", True),
    "stuck":            ("Stuck sensor", "One sensor freezes at a constant value.", True),
    "noise":            ("Noisy sensor", "One sensor becomes erratic.", True),
    "drift":            ("Drifting sensor", "One sensor slowly walks away from the truth.", True),
    "smoke_stuck":      ("Smoke event + stuck sensor", "A real smoke event while one sensor is frozen.", True),
}
INJECT_AT = PREFILL + 20        # incident begins shortly after live analysis starts


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["pm25", "pm10", "co", "voc"]:
        df[c] = df[c].clip(lower=0)
    df["humidity"] = df["humidity"].clip(0, 100)
    return df.round(3)


def build_timeline(scenario: str, sensor: str | None, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = PREFILL + SIM_STEPS_AFTER_PREFILL
    df = baseline(n, rng)
    truth = pd.DataFrame({"event": "none", "fault": "none", "faulty_sensor": "none"}, index=range(n))

    def mark_event(s, e, kind):
        truth.loc[s:e - 1, "event"] = kind

    def mark_fault(s, e, rows, kind, sen):
        for i in (rows if rows is not None else range(s, e)):
            truth.loc[i, ["fault", "faulty_sensor"]] = [kind, sen]

    if scenario in ("smoke_fire", "gas_leak", "poor_air_quality"):
        s, e = inject_event(df, INJECT_AT, scenario, rng)
        mark_event(s, e, scenario)
    elif scenario in ("spike", "stuck", "noise", "drift"):
        sen = sensor or str(rng.choice(SENSORS))
        s, e, rows = inject_fault(df, INJECT_AT, scenario, sen, rng)
        mark_fault(s, e, rows, scenario, sen)
    elif scenario == "smoke_stuck":
        sen = sensor or "pm25"
        s, e = inject_event(df, INJECT_AT, "smoke_fire", rng)
        mark_event(s, e, "smoke_fire")
        fs, fe, rows = inject_fault(df, INJECT_AT, "stuck", sen, rng)
        mark_fault(fs, fe, rows, "stuck", sen)
    elif scenario != "normal":
        raise ValueError(f"unknown scenario: {scenario}")

    return _finalize(df), truth


class Simulation:
    def __init__(self, scenario: str, sensor: str | None = None, seed: int | None = None):
        self.scenario, self.sensor = scenario, sensor
        self.df, self.truth = build_timeline(scenario, sensor, np.random.default_rng(seed))
        self.pos = PREFILL

    def prefill(self) -> pd.DataFrame:
        return self.df.iloc[:PREFILL][SENSORS]

    @property
    def finished(self) -> bool:
        return self.pos >= len(self.df)

    def next(self) -> tuple[dict, dict]:
        row = self.df.iloc[self.pos][SENSORS].to_dict()
        truth = self.truth.iloc[self.pos].to_dict()
        self.pos += 1
        return {k: float(v) for k, v in row.items()}, {k: str(v) for k, v in truth.items()}


def healthy_prefill(rng=None) -> pd.DataFrame:
    """Healthy history used when the API starts or is reset, so manual analysis works at once."""
    rng = rng or np.random.default_rng()
    return _finalize(baseline(PREFILL, rng))[SENSORS]