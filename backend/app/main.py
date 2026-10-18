"""
Step 8: FastAPI backend.

Run from the sensora/ root:
    uvicorn backend.app.main:app --reload
Then open  http://localhost:8000/docs  to try every endpoint in the browser.
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import CORS_ORIGINS
from backend.app.schemas import (AnalysisResponse, AnalyzeRequest, ScenarioInfo,
                                 SimStartRequest)
from backend.app.services.inference import Engine
from backend.app.services.monitor import Monitor
from backend.app.services.simulator import SCENARIOS

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = Engine()                      # models are loaded once, at startup
    state["engine"] = engine
    state["monitor"] = Monitor(engine)
    yield
    state.clear()


app = FastAPI(
    title="Sensora - Sensor Trust, Drift and False-Alarm Detection",
    description="Distinguishes genuine environmental events from sensor faults, "
                "scores sensor trust and explains every decision.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])


def monitor() -> Monitor:
    return state["monitor"]


# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": "engine" in state, "step": monitor().step}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(req: AnalyzeRequest):
    """Analyse one set of manually entered readings (appended to the live buffer)."""
    return monitor().ingest(req.readings.model_dump())


@app.get("/simulate/scenarios", response_model=list[ScenarioInfo])
def scenarios():
    return [ScenarioInfo(id=k, label=v[0], description=v[1], needs_sensor=v[2])
            for k, v in SCENARIOS.items()]


@app.post("/simulate/start")
def simulate_start(req: SimStartRequest):
    """Reset the monitor and prepare a scenario. Call /simulate/next repeatedly to play it."""
    return monitor().start_simulation(req.scenario, req.sensor)


@app.post("/simulate/next", response_model=AnalysisResponse)
def simulate_next():
    """Produce and analyse the next simulated reading (call every ~2 seconds)."""
    result = monitor().next_simulated()
    if result is None:
        raise HTTPException(status_code=409, detail="No simulation running. Call /simulate/start first.")
    return result


@app.get("/latest")
def latest():
    """Most recent reading in the buffer - use it to pre-fill a manual-entry form."""
    return monitor().latest()


@app.get("/history")
def history(limit: int = 50):
    return list(monitor().history)[-limit:]


@app.get("/trust")
def trust():
    return monitor().trust.snapshot()


@app.post("/reset")
def reset():
    monitor().reset()
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# Production: serve the built React app (frontend/dist) from the same server.
# Must stay LAST so it never shadows the API routes above.
DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")