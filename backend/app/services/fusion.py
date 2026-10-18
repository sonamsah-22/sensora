"""
Step 5: Fusion - the decision layer.

Combines:
  * the fault model  (which sensors look broken?)
  * the event model  (does the environment look hazardous?)
  * cross-sensor evidence (do OTHER, healthy sensors agree?)

into one decision and an alarm action. Pure rules on top of model outputs:
easy to unit-test, easy to explain, tunable without retraining.

Decisions
  NORMAL                       nothing wrong
  GENUINE_EVENT                real event, corroborated by several healthy sensors
  GENUINE_EVENT_DEGRADED       real event, but some sensor(s) in it are faulty
  SENSOR_FAULT                 anomaly explained by a faulty sensor -> no hazard
  NEEDS_VERIFICATION           evidence is weak or uncorroborated -> check manually

Alarm actions: TRIGGER_ALARM | SUPPRESS_ALARM | VERIFY | NONE
"""
FAULT_THR = 0.50      # P(any fault) needed to call a sensor faulty
EVENT_THR = 0.50      # P(any event) needed to call an event
UNSURE_LO = 0.40      # between UNSURE_LO and the threshold -> ask for verification
DEV_THR = 3.0         # |zlevel| above this = sensor deviating from its baseline
MIN_SUPPORT = 2       # healthy deviating sensors needed to corroborate an event
COMMON_CAUSE_EVENT = 0.90   # event confidence needed for the common-cause override
COMMON_CAUSE_SENSORS = 3    # sensors deviating together that a single fault cannot explain


def _top(probs: dict) -> tuple[str, float]:
    """Most likely non-'none' class and P(anything but 'none')."""
    label = max((c for c in probs if c != "none"), key=probs.get)
    return label, 1.0 - probs["none"]


def fuse(raw: dict) -> dict:
    # ---- sensor health ------------------------------------------------
    faulty, unsure_faulty = {}, {}
    for s, p in raw["fault_probs"].items():
        label, p_fault = _top(p)
        if p_fault >= FAULT_THR:
            faulty[s] = {"type": label, "confidence": round(p_fault, 3)}
        elif p_fault >= UNSURE_LO:
            unsure_faulty[s] = {"type": label, "confidence": round(p_fault, 3)}

    # ---- environment --------------------------------------------------
    ev_label, p_event = _top(raw["event_probs"])
    event_flag = p_event >= EVENT_THR
    z = raw["zlevel"]
    deviating = [s for s in z if abs(z[s]) > DEV_THR]
    support = [s for s in deviating if s not in faulty]   # healthy sensors that agree

    reasons = []
    event = None
    common_cause = False

    # ---- decision -----------------------------------------------------
    if event_flag:
        event = {"type": ev_label, "confidence": round(p_event, 3)}
        if not faulty:
            if len(support) >= MIN_SUPPORT:
                decision, action = "GENUINE_EVENT", "TRIGGER_ALARM"
                reasons.append(f"Model detected {ev_label} (confidence {p_event:.0%}).")
                reasons.append(f"{len(support)} independent sensors agree: {', '.join(support)}.")
            else:
                decision, action = "NEEDS_VERIFICATION", "VERIFY"
                reasons.append(f"Model suggests {ev_label} ({p_event:.0%}), but only "
                               f"{len(support)} sensor(s) deviate - not enough to corroborate.")
        elif len(support) >= MIN_SUPPORT:
            decision, action = "GENUINE_EVENT_DEGRADED", "TRIGGER_ALARM"
            reasons.append(f"Model detected {ev_label} (confidence {p_event:.0%}).")
            reasons.append(f"Still corroborated by {len(support)} healthy sensors: {', '.join(support)}.")
            for s, v in faulty.items():
                reasons.append(f"{s} shows a {v['type']} fault and is excluded from the evidence.")
        elif p_event >= COMMON_CAUSE_EVENT and len(deviating) >= COMMON_CAUSE_SENSORS:
            # Common-cause rule: the event model is very sure and several sensors moved together.
            # Several INDEPENDENT sensors failing in step with a recognised event pattern is far less
            # likely than the fault flags being wrong (a steep real ramp can mimic noise).
            decision, action = "GENUINE_EVENT_DEGRADED", "TRIGGER_ALARM"
            common_cause = True
            support = list(deviating)
            reasons.append(f"Model detected {ev_label} (confidence {p_event:.0%}) and {len(deviating)} sensors "
                           f"moved together ({', '.join(deviating)}).")
            reasons.append("Some of these were flagged as faulty, but simultaneous independent faults in step with "
                           "a known event pattern are unlikely - the fault flags are treated as unreliable.")
        elif len(support) >= 1:
            # Safety rule: never silently dismiss a possible hazard. The event model fired and at
            # least one healthy sensor agrees, but too little evidence is left after excluding the
            # faulty sensor(s) -> ask for a manual check instead of suppressing.
            decision, action = "NEEDS_VERIFICATION", "VERIFY"
            reasons.append(f"Possible {ev_label} ({p_event:.0%}), but the strongest sensor(s) are faulty and "
                           f"only {len(support)} healthy sensor agrees - verify before dismissing.")
            for s, v in faulty.items():
                reasons.append(f"{s}: {v['type']} fault (confidence {v['confidence']:.0%}).")
        else:
            decision, action = "SENSOR_FAULT", "SUPPRESS_ALARM"
            reasons.append(f"Event signal ({ev_label}) is not corroborated by any healthy sensor once "
                           "faulty sensors are removed - it is most likely produced by the fault.")
            for s, v in faulty.items():
                reasons.append(f"{s}: {v['type']} fault (confidence {v['confidence']:.0%}).")
    elif faulty:
        decision, action = "SENSOR_FAULT", "SUPPRESS_ALARM"
        for s, v in faulty.items():
            reasons.append(f"{s}: {v['type']} fault (confidence {v['confidence']:.0%}).")
        reasons.append("No other sensor shows a matching change, so this is not an "
                       "environmental event." if len(support) < MIN_SUPPORT else
                       "Other sensors are deviating but the event model does not recognise a known event.")
    elif p_event >= UNSURE_LO or unsure_faulty:
        decision, action = "NEEDS_VERIFICATION", "VERIFY"
        if p_event >= UNSURE_LO:
            reasons.append(f"Weak event signal: {ev_label} at {p_event:.0%}.")
        for s, v in unsure_faulty.items():
            reasons.append(f"{s}: possible {v['type']} fault ({v['confidence']:.0%}).")
    else:
        decision, action = "NORMAL", "NONE"
        reasons.append("All sensors behave consistently with their recent baseline.")

    return {
        "decision": decision,
        "alarm_action": action,
        "event": event,
        "faulty_sensors": faulty,
        "possible_faults": unsure_faulty,
        "corroborating_sensors": support if event_flag else [],
        "common_cause": common_cause,
        "deviating_sensors": deviating,
        "reasons": reasons,
    }