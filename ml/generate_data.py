"""
Step 1: Synthetic sensor stream generator.

Produces ONE continuous time series (not shuffled rows) with:
  - 7 correlated sensors with realistic baselines
  - Real environmental events (multi-sensor, coherent)
  - Injected sensor faults (single-sensor): spike, stuck, noise, drift
  - Ground-truth labels for every row

Run:  python generate_data.py
Out:  data/stream.csv
"""
import os
import numpy as np
import pandas as pd

SENSORS = ["pm25", "pm10", "co", "co2", "voc", "temp", "humidity"]

# Typical "size" of each sensor's normal variation; used to scale fault magnitudes
SCALE = {"pm25": 3.0, "pm10": 5.0, "co": 0.1, "co2": 40.0,
         "voc": 20.0, "temp": 1.0, "humidity": 3.0}

# Real events: which sensors move together, and by how much (peak amplitude)
EVENTS = {
    "smoke_fire":       {"pm25": 150, "pm10": 210, "co": 12, "voc": 300,
                         "temp": 6, "humidity": -12, "co2": 350},
    "gas_leak":         {"voc": 600, "co": 3, "co2": 120},
    "poor_air_quality": {"pm25": 60, "pm10": 90, "co2": 250, "voc": 60},
}
FAULTS = ["spike", "stuck", "noise", "drift"]


def ar1(n, phi, sigma, rng):
    """Autocorrelated noise: each value depends on the previous one."""
    x = np.zeros(n)
    e = rng.normal(0, sigma, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def baseline(n, rng):
    t = np.arange(n)
    day = 2 * np.pi * t / 1440  # slow daily cycle
    pm25 = np.clip(12 + ar1(n, 0.95, 0.6, rng), 1, None)
    pm10 = np.clip(4 + 1.6 * pm25 + ar1(n, 0.9, 0.8, rng), 1, None)
    co = np.clip(0.4 + ar1(n, 0.95, 0.02, rng), 0.05, None)
    co2 = 480 + 40 * np.sin(day) + ar1(n, 0.97, 6, rng)
    voc = np.clip(90 + ar1(n, 0.95, 4, rng), 5, None)
    temp = 26 + 2 * np.sin(day) + ar1(n, 0.98, 0.05, rng)
    humidity = np.clip(55 - 2.5 * (temp - 26) + ar1(n, 0.95, 0.4, rng), 5, 99)
    return pd.DataFrame(dict(pm25=pm25, pm10=pm10, co=co, co2=co2,
                             voc=voc, temp=temp, humidity=humidity))


def envelope(length, rng):
    """Ramp up -> hold -> decay, values 0..1."""
    up = max(3, length // 4)
    down = max(3, length // 3)
    hold = max(1, length - up - down)
    return np.concatenate([np.linspace(0, 1, up),
                           np.ones(hold),
                           np.linspace(1, 0, down)])


def inject_event(df, start, kind, rng):
    length = int(rng.integers(40, 90))
    env = envelope(length, rng)
    length = len(env)
    end = min(start + length, len(df))
    env = env[: end - start]
    scale = rng.uniform(0.6, 1.4)
    for sensor, amp in EVENTS[kind].items():
        df.loc[start:end - 1, sensor] += amp * scale * env
    return start, end


def wide(rng, lo, hi):
    """Log-uniform magnitude: small AND huge faults are both common.
    Real failing sensors often read absurdly high/low, so a fault must NOT be
    recognisable merely by being small - the model has to use cross-sensor evidence."""
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def inject_fault(df, start, kind, sensor, rng):
    s = SCALE[sensor]
    if kind == "spike":
        end = min(start + 30, len(df))
        idx = rng.choice(np.arange(start, end), size=int(rng.integers(1, 4)), replace=False)
        for i in idx:
            df.loc[i, sensor] += rng.choice([-1, 1]) * s * wide(rng, 6, 80)
        return start, end, list(idx)
    length = int(rng.integers(30, 80)) if kind != "drift" else int(rng.integers(100, 250))
    end = min(start + length, len(df))
    if kind == "stuck":
        value = df.loc[start, sensor]
        if rng.random() < 0.25:        # sensor latches at a wrong, extreme value
            value += rng.choice([-1, 1]) * s * wide(rng, 8, 40)
        df.loc[start:end - 1, sensor] = value
    elif kind == "noise":
        df.loc[start:end - 1, sensor] += rng.normal(0, s * wide(rng, 1.0, 5.0), end - start)
    elif kind == "drift":
        bias = rng.choice([-1, 1]) * s * wide(rng, 5, 40)
        ramp = np.linspace(0, bias, end - start)
        df.loc[start:end - 1, sensor] += ramp
    return start, end, None


def generate(n=12000, seed=42):
    rng = np.random.default_rng(seed)
    df = baseline(n, rng)
    df["event"] = "none"
    df["fault"] = "none"
    df["faulty_sensor"] = "none"

    cursor = int(rng.integers(100, 200))
    ev_i, f_i = 0, 0  # round-robin so every class gets examples
    while cursor < n - 300:
        if rng.random() < 0.4:  # real event
            kind = list(EVENTS)[ev_i % len(EVENTS)]; ev_i += 1
            s, e = inject_event(df, cursor, kind, rng)
            df.loc[s:e - 1, "event"] = kind
            # ~35% of real events ALSO have a broken sensor (the hard case:
            # model must not assume "other sensors deviating => no fault")
            if rng.random() < 0.35:
                fkind = FAULTS[f_i % len(FAULTS)]; f_i += 1
                fsensor = rng.choice(SENSORS)
                fstart = s + int(rng.integers(0, max(1, (e - s) // 2)))
                fs, fe, rows = inject_fault(df, fstart, fkind, fsensor, rng)
                if rows is None:
                    rows = range(fs, fe)
                for i in rows:
                    df.loc[i, "fault"] = fkind
                    df.loc[i, "faulty_sensor"] = fsensor
                e = max(e, fe)
            cursor = e
        else:                   # sensor fault
            kind = FAULTS[f_i % len(FAULTS)]; f_i += 1
            sensor = rng.choice(SENSORS)
            s, e, rows = inject_fault(df, cursor, kind, sensor, rng)
            if rows is None:
                rows = range(s, e)
            for i in rows:
                df.loc[i, "fault"] = kind
                df.loc[i, "faulty_sensor"] = sensor
            cursor = e
        cursor += int(rng.integers(100, 250))  # normal gap

    # keep readings physically plausible
    for c in ["pm25", "pm10", "co", "voc"]:
        df[c] = df[c].clip(lower=0)
    df["humidity"] = df["humidity"].clip(0, 100)
    df.insert(0, "t", np.arange(n))
    return df.round(3)


from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # sensora/
DATA_DIR = ROOT / "data"

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    data = generate()
    out = DATA_DIR / "stream.csv"
    data.to_csv(out, index=False)
    print("Saved", out, data.shape)