"""
Step 2: Feature engineering.

Turns a raw stream (one row per time step, 7 sensor columns) into features
that describe the RECENT BEHAVIOUR of each sensor and how the sensors agree
with each other.

Every feature uses only past + current rows (trailing windows), so the exact
same function works offline for training and online for the live backend:
    keep the last WARMUP+ readings in a buffer -> compute_features(buffer) -> take last row.

Run a sanity check from the sensora/ root:
    python -m ml.features
"""
import numpy as np
import pandas as pd

from ml.common import (SENSORS, SCALE, WINDOW_SHORT as W, WINDOW_LONG as L,
                       WINDOW_BASE as B, WARMUP, DEV_THRESHOLD, PAIRS, DATA_DIR)

# Features describing ONE sensor (used by the fault model, long format)
SENSOR_FEATURES = ["std", "dstd", "zero_frac", "maxjump",
                   "slope_s", "slope_l", "zlevel", "zbase", "others_dev", "resid"]


def rolling_slope(x: np.ndarray, w: int) -> np.ndarray:
    """Least-squares slope over a trailing window of w points (per step)."""
    k = np.arange(w) - (w - 1) / 2
    kernel = (k / (k ** 2).sum())[::-1]
    out = np.convolve(x, kernel, mode="full")[: len(x)]
    out[: w - 1] = 0.0
    return out


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Wide feature table: one row per time step, columns like 'pm25__std'."""
    f = pd.DataFrame(index=df.index)

    # ---- 1. per-sensor behaviour -------------------------------------
    for s in SENSORS:
        x, sc = df[s], SCALE[s]
        d = x.diff()
        f[f"{s}__std"] = x.rolling(W).std() / sc                 # spread -> noise
        f[f"{s}__dstd"] = d.rolling(W).std() / sc                # jitter  -> noise
        f[f"{s}__zero_frac"] = (d.abs() < 1e-9).rolling(20).mean()  # frozen -> stuck-at
        f[f"{s}__maxjump"] = d.abs().rolling(W).max() / sc       # biggest step -> spike
        f[f"{s}__slope_s"] = rolling_slope(x.values, W) * W / sc  # change over 30 steps
        f[f"{s}__slope_l"] = rolling_slope(x.values, L) * L / sc  # change over 120 -> drift
        f[f"{s}__zlevel"] = (x.rolling(5).mean() - x.rolling(L).median()) / sc  # level vs recent baseline
        f[f"{s}__zbase"] = (x.rolling(5).mean() - x.rolling(B).median()) / sc   # level vs long baseline -> drift

    # ---- 2. cross-sensor agreement -----------------------------------
    dev = pd.DataFrame({s: f[f"{s}__zlevel"].abs() > DEV_THRESHOLD for s in SENSORS})
    n_dev = dev.sum(axis=1)
    f["n_dev"] = n_dev                                  # how many sensors are off baseline
    for s in SENSORS:
        f[f"{s}__others_dev"] = n_dev - dev[s]          # how many OTHERS agree with it

    # correlation is undefined when a sensor is frozen (zero variance) -> 0, never inf
    f["pm_corr"] = df["pm25"].rolling(W).corr(df["pm10"]).replace([np.inf, -np.inf], np.nan).clip(-1, 1)
    f["th_corr"] = df["temp"].rolling(W).corr(df["humidity"]).replace([np.inf, -np.inf], np.nan).clip(-1, 1)
    # physical-consistency residuals: does the reading match what its neighbour implies?
    f["resid_pm"] = ((df["pm10"] - (4 + 1.6 * df["pm25"])) / SCALE["pm10"]).rolling(5).mean()
    f["resid_th"] = ((df["humidity"] - (55 - 2.5 * (df["temp"] - 26))) / SCALE["humidity"]).rolling(5).mean()
    for s in SENSORS:
        f[f"{s}__resid"] = f[PAIRS[s]] if s in PAIRS else 0.0

    return f.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ---------------------------------------------------------------------------
# Model-ready views
# ---------------------------------------------------------------------------
def fault_long(f: pd.DataFrame) -> pd.DataFrame:
    """Stack the 7 sensors into one table: one row per (time, sensor).
    A single shared model then learns what 'stuck' or 'drifting' looks like
    for ANY sensor, with 7x the training data."""
    parts = []
    for s in SENSORS:
        p = f[[f"{s}__{c}" for c in SENSOR_FEATURES]].copy()
        p.columns = SENSOR_FEATURES
        p["sensor"] = s
        p["t"] = f.index
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def event_features(f: pd.DataFrame) -> pd.DataFrame:
    """Whole-environment view used by the event classifier."""
    cols = [f"{s}__zlevel" for s in SENSORS] + [f"{s}__slope_s" for s in SENSORS]
    cols += ["n_dev", "pm_corr", "th_corr", "resid_pm", "resid_th"]
    return f[cols]


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def fault_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Per-sensor fault label for every time step (columns = sensors).
    A spike is a single-step event, so its label is held for W steps -
    exactly as long as it stays visible inside the feature window."""
    lab = pd.DataFrame("none", index=df.index, columns=SENSORS)
    for s in SENSORS:
        m = df["faulty_sensor"] == s
        lab.loc[m, s] = df.loc[m, "fault"]
        recent_spike = (lab[s] == "spike").astype(int).rolling(W, min_periods=1).max() > 0
        lab.loc[recent_spike & (lab[s] == "none"), s] = "spike"
    return lab


def fault_long_labels(lab: pd.DataFrame) -> pd.Series:
    """Labels in the same row order as fault_long()."""
    return pd.concat([lab[s] for s in SENSORS], ignore_index=True)


def chrono_split(n: int, train=0.70, val=0.15):
    """Time-ordered split. Never shuffle a time series (leakage)."""
    a, b = int(n * train), int(n * (train + val))
    return slice(0, a), slice(a, b), slice(b, n)


# ---------------------------------------------------------------------------
# Sanity check: do the features actually separate the cases?
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "stream.csv")
    f = compute_features(df)
    lab = fault_labels(df)
    print("features:", f.shape, "| warm-up rows to drop:", WARMUP)

    long_x = fault_long(f)
    long_y = fault_long_labels(lab)
    long_x = long_x[long_x["t"] >= WARMUP]
    long_y = long_y.loc[long_x.index]
    print("fault table:", long_x.shape)
    print(long_y.value_counts().to_string(), "\n")

    # Mean of the 'signature' feature for each fault type vs normal
    sig = {"spike": "maxjump", "stuck": "zero_frac", "noise": "dstd", "drift": "zbase"}
    print("Signature feature check (mean |value|, higher for its own fault = good):")
    for fault, col in sig.items():
        mine = long_x.loc[long_y == fault, col].abs().mean()
        norm = long_x.loc[long_y == "none", col].abs().mean()
        print(f"  {fault:6s} {col:10s} fault={mine:7.3f}  normal={norm:7.3f}  ratio={mine / max(norm, 1e-9):5.1f}x")

    print("\nCross-sensor check (mean n_dev = sensors off baseline):")
    ev = df["event"] != "none"
    fl = df["fault"] != "none"
    print(f"  during real events : {f.loc[ev, 'n_dev'].mean():.2f}")
    print(f"  during faults      : {f.loc[fl, 'n_dev'].mean():.2f}")
    print(f"  normal             : {f.loc[~ev & ~fl, 'n_dev'].mean():.2f}")