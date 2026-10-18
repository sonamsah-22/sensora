"""Shared constants and paths. Import from here, never redefine elsewhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # sensora/
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

SENSORS = ["pm25", "pm10", "co", "co2", "voc", "temp", "humidity"]

# "Datasheet" noise scale per sensor. Every feature is divided by this so the
# model sees unit-free numbers (a 2-unit jump means very different things for
# CO in ppm vs CO2 in ppm).
SCALE = {"pm25": 3.0, "pm10": 5.0, "co": 0.1, "co2": 40.0,
         "voc": 20.0, "temp": 1.0, "humidity": 3.0}

WINDOW_SHORT = 30     # catches spikes, stuck-at, noise
WINDOW_LONG = 120     # catches slow drift and gives a baseline level
WINDOW_BASE = 360     # long-term baseline; a drifting sensor can't hide in it
WARMUP = WINDOW_BASE  # first rows lack a full window; drop them in training
DEV_THRESHOLD = 3.0   # |zlevel| above this = "this sensor is deviating"

# Known physical relationships between neighbouring sensors
# (what a domain expert would tell the system)
PAIRS = {"pm25": "resid_pm", "pm10": "resid_pm",
         "temp": "resid_th", "humidity": "resid_th"}