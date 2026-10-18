"""
Step 6: Sensor trust score + maintenance recommendation.

Each sensor has a trust score from 0 to 100 that evolves over time:
  * it DROPS while the fault model flags the sensor (faster for nastier faults)
  * it RECOVERS slowly while the sensor behaves
  * real environmental events never reduce trust (only fault evidence does)

From the trust score and the recent fault history we also derive a
maintenance status: OK / MONITOR / INSPECT / RECALIBRATE / REPLACE.

Usage (one tracker per deployment; call update() once per new reading):
    tracker = TrustTracker()
    snapshot = tracker.update(raw["fault_probs"])      # raw from Engine.run()
"""
from collections import Counter, deque

from ml.common import SENSORS

# Trust points lost PER STEP while a fault is flagged (scaled by confidence).
# Spikes are brief, so they cost little per step; stuck/noise/drift persist.
STEP_PENALTY = {"spike": 0.5, "stuck": 1.5, "noise": 1.0, "drift": 0.8}
RECOVERY = 0.3          # trust points regained per healthy step (slow on purpose)
PENALTY_FROM = 0.40     # P(fault) at which trust starts to drop
FAULTY_AT = 0.50        # P(fault) at which a step is counted as "faulty" in history
HISTORY = 300           # steps of fault history used for maintenance decisions

OK_ABOVE = 85
MONITOR_ABOVE = 65
REPLACE_BELOW = 35
REPLACE_FAULTY_FRAC = 0.50   # faulty for more than half of recent history


class TrustTracker:
    def __init__(self, sensors=SENSORS):
        self.trust = {s: 100.0 for s in sensors}
        self.history = {s: deque(maxlen=HISTORY) for s in sensors}

    # ------------------------------------------------------------------
    def update(self, fault_probs: dict) -> dict:
        """fault_probs: {sensor: {class: probability}} for the newest reading."""
        for s, p in fault_probs.items():
            label = max((c for c in p if c != "none"), key=p.get)
            p_fault = 1.0 - p["none"]
            self.history[s].append(label if p_fault >= FAULTY_AT else "none")
            if p_fault >= PENALTY_FROM:
                self.trust[s] = max(0.0, self.trust[s] - STEP_PENALTY[label] * p_fault)
            else:
                self.trust[s] = min(100.0, self.trust[s] + RECOVERY)
        return self.snapshot()

    # ------------------------------------------------------------------
    def maintenance(self, s: str) -> dict:
        t = self.trust[s]
        faults = Counter(x for x in self.history[s] if x != "none")
        n = max(len(self.history[s]), 1)
        frac = sum(faults.values()) / n
        dominant = faults.most_common(1)[0][0] if faults else None

        if t >= OK_ABOVE:
            status, action = "OK", "No action needed."
            reason = ("Behaving normally." if not dominant else
                      f"Minor recent {dominant} activity, fully within tolerance.")
        elif t >= MONITOR_ABOVE:
            status, action = "MONITOR", "Keep under observation; no maintenance yet."
            reason = f"Recent {dominant or 'irregular'} behaviour has lowered trust."
        elif dominant == "drift":
            status, action = "RECALIBRATE", "Recalibrate against a reference instrument."
            reason = "Persistent drift: readings are slowly moving away from the true value."
        elif t < REPLACE_BELOW or frac > REPLACE_FAULTY_FRAC:
            status, action = "REPLACE", "Replace the sensor."
            reason = (f"Sensor has been faulty {frac:.0%} of the recent period "
                      f"(mostly {dominant}); trust is {t:.0f}/100.")
        elif dominant == "spike":
            status, action = "INSPECT", "Check wiring, power supply and electrical interference."
            reason = "Repeated sudden spikes usually point to interference or a loose connection."
        else:
            status, action = "INSPECT", "Inspect the sensor on site."
            reason = f"Recurring {dominant or 'unreliable'} readings; trust is {t:.0f}/100."

        return {"status": status, "action": action, "reason": reason,
                "dominant_fault": dominant, "faulty_fraction": round(frac, 3)}

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        sensors = {s: {"trust": round(self.trust[s], 1), **self.maintenance(s)}
                   for s in self.trust}
        avg = sum(v["trust"] for v in sensors.values()) / len(sensors)
        return {"average_trust": round(avg, 1), "sensors": sensors}