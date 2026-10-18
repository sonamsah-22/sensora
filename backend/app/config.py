"""Backend settings. Model/feature constants live in ml/common.py (single source of truth)."""
import os

from ml.common import WARMUP

BUFFER_SIZE = 500          # readings kept in memory for window features
PREFILL = 400              # healthy history loaded at start so analysis works immediately
HISTORY_SIZE = 100         # recent analyses returned by GET /history
SIM_STEPS_AFTER_PREFILL = 500

CORS_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",   # Vite (React) dev server
    "http://localhost:3000", "http://127.0.0.1:3000",
] + [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]   # e.g. https://my-app.vercel.app

assert PREFILL >= WARMUP, "PREFILL must cover the feature warm-up window"