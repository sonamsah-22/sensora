"""
Step 7: Explainable decisions.

Turns the fusion result + the recent raw readings into plain-English evidence:
what was seen, why it was called a fault or a real event, and what to do.

Every sentence is built from MEASUREMENTS taken from the buffer (e.g. "identical
in 20 of the last 20 readings"), not from the model's internals, so the
explanation is always something an operator can verify.

    result = explain(fused, raw, buffer, trust_snapshot)
"""
import numpy as np
import pandas as pd

from ml.common import SCALE, WINDOW_SHORT, WINDOW_BASE

NAMES = {"pm25": "PM2.5", "pm10": "PM10", "co": "Carbon monoxide", "co2": "Carbon dioxide",
         "voc": "VOC", "temp": "Temperature", "humidity": "Humidity"}
UNITS = {"pm25": "µg/m³", "pm10": "µg/m³", "co": "ppm", "co2": "ppm",
         "voc": "ppb", "temp": "°C", "humidity": "%"}

EVENT_NAMES = {"smoke_fire": "smoke / fire", "gas_leak": "gas leak",
               "poor_air_quality": "poor air quality"}
EVENT_ACTIONS = {
    "smoke_fire": ["Verify visually or with a second detector.",
                   "Follow the fire-response protocol and evacuate if confirmed."],
    "gas_leak": ["Ventilate the area and isolate the suspected source.",
                 "Keep ignition sources away until a handheld meter confirms the level."],
    "poor_air_quality": ["Increase ventilation or filtration.",
                         "Limit exposure for sensitive people until readings fall."],
}


def _fmt(sensor, v):
    return f"{v:.2f}".rstrip("0").rstrip(".") + f" {UNITS[sensor]}"


# ---------------------------------------------------------------------------
# Measured evidence per sensor
# ---------------------------------------------------------------------------
def _baseline(x: pd.Series) -> float:
    return float(x.tail(WINDOW_BASE).median())


def _fault_evidence(sensor: str, fault_type: str, x: pd.Series) -> str:
    name, sc = NAMES[sensor], SCALE[sensor]
    recent = x.tail(WINDOW_SHORT)

    if fault_type == "stuck":
        last20 = x.tail(21).diff().dropna()
        same = int((last20.abs() < 1e-9).sum())
        return (f"{name} is frozen at {_fmt(sensor, x.iloc[-1])} - the value was identical "
                f"in {same} of the last 20 readings, while a working sensor always fluctuates slightly.")

    if fault_type == "spike":
        d = recent.diff().abs().fillna(0.0)
        k = int(np.argmax(d.values))
        jump = float(d.iloc[k])
        ago = len(recent) - 1 - k
        when = "in the latest reading" if ago == 0 else f"{ago} readings ago"
        return (f"{name} made a sudden jump of {_fmt(sensor, jump)} in a single reading ({when}), "
                f"about {jump / sc:.0f}x its normal variation - too abrupt to be a real environmental change.")

    if fault_type == "noise":
        recent_j = float(recent.diff().std())
        older = x.iloc[-WINDOW_BASE:-WINDOW_SHORT].diff().std() if len(x) > WINDOW_SHORT + 20 else np.nan
        ref = float(older) if older == older and older > 0 else sc * 0.3
        return (f"{name} readings are erratic: step-to-step jitter is {recent_j / ref:.1f}x its usual level, "
                f"which points to electrical noise or a failing element.")

    if fault_type == "drift":
        dev = float(x.tail(5).mean() - _baseline(x))
        direction = "above" if dev > 0 else "below"
        return (f"{name} has moved {_fmt(sensor, abs(dev))} {direction} its long-term baseline "
                f"({abs(dev) / sc:.1f}x normal variation) without matching changes in other sensors - "
                f"typical of calibration drift.")

    return f"{name} shows abnormal behaviour ({fault_type})."


def _event_evidence(sensor: str, x: pd.Series) -> str:
    name = NAMES[sensor]
    dev = float(x.tail(5).mean() - _baseline(x))
    verb = "rose" if dev > 0 else "fell"
    return f"{name} {verb} by {_fmt(sensor, abs(dev))} from its baseline ({abs(dev) / SCALE[sensor]:.1f}x normal variation)."


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def explain(fused: dict, raw: dict, buffer: pd.DataFrame, trust: dict | None = None) -> dict:
    decision = fused["decision"]
    event = fused["event"]
    faulty = fused["faulty_sensors"]
    support = fused["corroborating_sensors"]
    ev_name = EVENT_NAMES.get(event["type"], event["type"]) if event else None
    fnames = ", ".join(NAMES[s] for s in faulty)

    evidence, why, actions = [], [], []

    # ---- evidence from the measurements ------------------------------
    for s, f in faulty.items():
        evidence.append({"sensor": s, "kind": "fault", "confidence": f["confidence"],
                         "text": _fault_evidence(s, f["type"], buffer[s])})
    if decision == "NEEDS_VERIFICATION":
        for s, f in fused["possible_faults"].items():
            evidence.append({"sensor": s, "kind": "possible_fault", "confidence": f["confidence"],
                             "text": _fault_evidence(s, f["type"], buffer[s])})
    for s in support:
        evidence.append({"sensor": s, "kind": "event", "confidence": None,
                         "text": _event_evidence(s, buffer[s])})

    # ---- headline, reasoning chain, actions --------------------------
    if decision == "NORMAL":
        severity = "info"
        headline = "All sensors normal"
        summary = "Every sensor is behaving consistently with its recent baseline."
        why = ["No sensor shows spike, freeze, noise or drift patterns.",
               "No coordinated change across sensors was found."]
        actions = ["No action needed."]

    elif decision == "GENUINE_EVENT":
        severity = "critical"
        headline = f"Genuine event: {ev_name}"
        summary = f"{len(support)} independent sensors agree on a {ev_name} pattern. Trust this alarm."
        why = [f"The event model recognised a {ev_name} pattern ({event['confidence']:.0%} confidence).",
               f"Several different sensors changed together ({', '.join(NAMES[s] for s in support)}); "
               "a single faulty sensor cannot cause that.",
               "No sensor shows a fault pattern."]
        actions = EVENT_ACTIONS.get(event["type"], ["Investigate the affected area."])

    elif decision == "GENUINE_EVENT_DEGRADED" and fused.get("common_cause"):
        severity = "critical"
        headline = f"Genuine event: {ev_name}"
        summary = (f"{len(support)} sensors moved together in a {ev_name} pattern. Some were flagged as "
                   "faulty, but the flags are overridden - treat this alarm as real.")
        why = [f"The event model recognised a {ev_name} pattern ({event['confidence']:.0%} confidence).",
               f"{len(support)} sensors changed in step ({', '.join(NAMES[s] for s in support)}).",
               "Several independent sensors failing at the same moment, in line with a known event, is far less "
               "likely than the fault flags being false (a steep real ramp can resemble noise)."]
        actions = list(EVENT_ACTIONS.get(event["type"], ["Investigate the affected area."]))
        actions.append("Re-check the flagged sensors once the situation is under control.")

    elif decision == "GENUINE_EVENT_DEGRADED":
        severity = "critical"
        headline = f"Genuine event: {ev_name} (with unreliable sensor)"
        summary = (f"A {ev_name} is confirmed by {len(support)} healthy sensors, but {fnames} "
                   "is faulty and was excluded from the evidence.")
        why = [f"The event model recognised a {ev_name} pattern ({event['confidence']:.0%} confidence).",
               f"It is still corroborated by healthy sensors: {', '.join(NAMES[s] for s in support)}.",
               f"{fnames} shows a fault pattern, so its readings were not counted."]
        actions = EVENT_ACTIONS.get(event["type"], ["Investigate the affected area."])
        actions.append(f"Do not rely on {fnames} until it has been checked.")

    elif decision == "SENSOR_FAULT":
        severity = "warning"
        headline = f"Sensor fault: {fnames} - not an environmental event"
        summary = "The anomaly is explained by a faulty sensor. The alarm is suppressed."
        if support:
            why.append("Other sensors deviate, but they do not form a recognised event pattern.")
        else:
            why.append("No other sensor shows a matching change, so the environment itself looks unchanged.")
        if event:
            why.append(f"A {ev_name} signal was seen, but it disappears once the faulty sensor is removed.")
        why.append("The faulty sensor's own behaviour matches a known fault signature (see evidence).")
        actions = ["No hazard response is needed on this evidence."]

    else:  # NEEDS_VERIFICATION
        severity = "warning"
        headline = "Unclear situation - verify manually"
        summary = "The evidence is weak or not corroborated, so the system will not raise or dismiss an alarm."
        for s, f in fused["possible_faults"].items():
            why.append(f"{NAMES[s]} may have a {f['type']} fault ({f['confidence']:.0%} confidence) - "
                       "suggestive, but not strong enough to act on.")
        why += [r for r in fused["reasons"] if r.startswith(("Weak event", "Model suggests"))]
        if not why:
            why = list(fused["reasons"])
        actions = ["Check the area or the flagged sensor with a handheld reference instrument."]

    # ---- maintenance advice for faulty sensors -----------------------
    if trust:
        for s in faulty:
            m = trust["sensors"].get(s)
            if m and m["status"] in ("INSPECT", "RECALIBRATE", "REPLACE"):
                actions.append(f"{NAMES[s]}: {m['action']} (trust {m['trust']:.0f}/100)")

    return {"headline": headline, "severity": severity, "summary": summary,
            "evidence": evidence, "why": why, "recommended_actions": actions}