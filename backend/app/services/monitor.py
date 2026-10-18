"""
The Monitor ties everything together for one deployment:

    new reading -> rolling buffer -> models (inference) -> fusion -> trust -> explanation

It owns the state that must persist between requests: the buffer of recent
readings, the trust scores, the analysis history and the running simulation.
"""
from collections import deque
from datetime import datetime, timezone
from threading import RLock

import pandas as pd

from backend.app.config import BUFFER_SIZE, HISTORY_SIZE
from backend.app.services.explain import explain
from backend.app.services.fusion import fuse
from backend.app.services.inference import Engine
from backend.app.services.simulator import Simulation, healthy_prefill
from backend.app.services.trust import TrustTracker
from ml.common import SENSORS


SMOOTH_STEPS = 3     # model outputs are averaged over the last 3 readings


def smooth(recent: list[dict]) -> dict:
    """Average the model probabilities over recent readings so a one-step flicker
    cannot flip a decision. Level features (zlevel, n_dev) use the newest reading."""
    n = len(recent)
    out = dict(recent[-1])
    out["fault_probs"] = {
        s: {c: sum(r["fault_probs"][s][c] for r in recent) / n for c in recent[-1]["fault_probs"][s]}
        for s in recent[-1]["fault_probs"]}
    out["event_probs"] = {c: sum(r["event_probs"][c] for r in recent) / n for c in recent[-1]["event_probs"]}
    return out


class Monitor:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.lock = RLock()
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, prefill: pd.DataFrame | None = None, sim: Simulation | None = None):
        with self.lock:
            if prefill is None:
                prefill = healthy_prefill()
            self.buffer = deque(prefill[SENSORS].values.tolist(), maxlen=BUFFER_SIZE)
            self.trust = TrustTracker()
            self.history = deque(maxlen=HISTORY_SIZE)
            self.recent_raw = deque(maxlen=SMOOTH_STEPS)
            self.step = 0
            self.sim = sim

    # ------------------------------------------------------------------
    def ingest(self, reading: dict, truth: dict | None = None, restarted: bool = False) -> dict:
        """Add one reading, analyse it, return the full result."""
        with self.lock:
            self.buffer.append([float(reading[s]) for s in SENSORS])
            df = pd.DataFrame(list(self.buffer), columns=SENSORS)

            self.recent_raw.append(self.engine.run_last(df))
            raw = smooth(list(self.recent_raw))
            fused = fuse(raw)
            trust = self.trust.update(raw["fault_probs"])
            explanation = explain(fused, raw, df, trust)
            self.step += 1

            result = {
                "step": self.step,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "readings": {s: float(reading[s]) for s in SENSORS},
                "decision": fused["decision"],
                "alarm_action": fused["alarm_action"],
                "event": fused["event"],
                "faulty_sensors": fused["faulty_sensors"],
                "possible_faults": fused["possible_faults"],
                "corroborating_sensors": fused["corroborating_sensors"],
                "trust": trust,
                "explanation": explanation,
                "truth": truth,
                "restarted": restarted,
            }
            self.history.append({
                "step": result["step"], "timestamp": result["timestamp"],
                "decision": result["decision"], "alarm_action": result["alarm_action"],
                "headline": explanation["headline"], "average_trust": trust["average_trust"],
                "truth": truth,
            })
            return result

    def latest(self) -> dict:
        """Most recent reading in the buffer (used to pre-fill the manual-entry form)."""
        with self.lock:
            return {"step": self.step, "readings": dict(zip(SENSORS, self.buffer[-1]))}

    # ------------------------------------------------------------------
    def start_simulation(self, scenario: str, sensor: str | None = None) -> dict:
        with self.lock:
            sim = Simulation(scenario, sensor)
            self.reset(prefill=sim.prefill(), sim=sim)
            return {"scenario": scenario, "sensor": sensor, "total_steps": len(sim.df) - len(sim.prefill())}

    def next_simulated(self) -> dict | None:
        """Advance the running simulation by one reading. None if no simulation is running."""
        with self.lock:
            if self.sim is None:
                return None
            restarted = False
            if self.sim.finished:                       # loop the scenario with fresh randomness
                scenario, sensor = self.sim.scenario, self.sim.sensor
                sim = Simulation(scenario, sensor)
                self.reset(prefill=sim.prefill(), sim=sim)
                restarted = True
            reading, truth = self.sim.next()
            return self.ingest(reading, truth=truth, restarted=restarted)