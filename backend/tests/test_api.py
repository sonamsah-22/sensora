"""
End-to-end API test: plays every scenario through the real endpoints.

Run from the sensora/ root:
    python -m backend.tests.test_api
(needs:  pip install httpx)
"""
import time
from collections import Counter

from fastapi.testclient import TestClient

from backend.app.main import app

STEPS = 140

# scenario -> decisions that are acceptable once the incident is underway
EXPECT_ALARM = {"smoke_fire", "gas_leak", "poor_air_quality", "smoke_stuck"}


def play(client, scenario, sensor=None, steps=STEPS):
    """Play a scenario and tally decisions against the GROUND TRUTH of each step."""
    client.post("/simulate/start", json={"scenario": scenario, "sensor": sensor})
    seen, timings, worst = Counter(), [], 100.0
    t = Counter()
    for _ in range(steps):
        t0 = time.time()
        r = client.post("/simulate/next").json()
        timings.append(time.time() - t0)
        seen[r["decision"]] += 1
        worst = min(worst, min(v["trust"] for v in r["trust"]["sensors"].values()))
        tr, act = r["truth"], r["alarm_action"]
        if tr["event"] != "none":
            t["event_steps"] += 1
            t["event_alarmed"] += act == "TRIGGER_ALARM"
            t["event_suppressed"] += act == "SUPPRESS_ALARM"
        elif tr["fault"] != "none":
            t["fault_steps"] += 1
            t["fault_false_alarm"] += act == "TRIGGER_ALARM"
        else:
            t["clear_steps"] += 1
            t["clear_flagged"] += r["decision"] != "NORMAL"
            t["clear_false_alarm"] += act == "TRIGGER_ALARM"
    return seen, sum(timings) / len(timings) * 1000, worst, t


if __name__ == "__main__":
    with TestClient(app) as client:
        print("GET /health ->", client.get("/health").json())
        scen = client.get("/simulate/scenarios").json()
        print("GET /simulate/scenarios ->", [s["id"] for s in scen], "\n")

        # manual analysis: start from the latest reading, then break ONE sensor
        base = client.get("/latest").json()["readings"]
        r = client.post("/analyze", json={"readings": base}).json()
        print("POST /analyze (latest, unchanged) ->", r["decision"], "|", r["explanation"]["headline"])
        r = client.post("/analyze", json={"readings": {**base, "voc": base["voc"] + 900}}).json()
        print("POST /analyze (only VOC +900)     ->", r["decision"], "|", r["explanation"]["headline"])
        bad = client.post("/analyze", json={"readings": {**base, "humidity": 140}})
        print("POST /analyze (humidity = 140)    ->", bad.status_code, "(input validation)")
        print()

        print(f"{'scenario':18s} {'avg ms':>7s} {'lowest trust':>13s}  decisions over {STEPS} steps")
        print("-" * 100)
        total = Counter()
        for sc in ["normal", "smoke_fire", "gas_leak", "poor_air_quality",
                   "spike", "stuck", "noise", "drift", "smoke_stuck"]:
            sensor = {"smoke_stuck": "pm25", "drift": "voc"}.get(sc)
            seen, ms, worst, t = play(client, sc, sensor)
            total.update(t)
            print(f"{sc:18s} {ms:7.0f} {worst:13.1f}  {dict(seen)}")
        print("-" * 100)
        print("SCORED AGAINST GROUND TRUTH (every step of every scenario):")
        print(f"  Real-event steps          : {total['event_steps']:4d}  -> alarm raised {total['event_alarmed']}, "
              f"SUPPRESSED {total['event_suppressed']}, other (verify/normal) "
              f"{total['event_steps'] - total['event_alarmed'] - total['event_suppressed']}")
        print(f"  Sensor-fault-only steps   : {total['fault_steps']:4d}  -> FALSE ALARMS {total['fault_false_alarm']}")
        print(f"  All-clear steps           : {total['clear_steps']:4d}  -> flagged anything {total['clear_flagged']} "
              f"({total['clear_flagged'] / max(total['clear_steps'], 1):.1%}), false alarms {total['clear_false_alarm']}")
        print("\nGET /history (last 3):")
        for h in client.get("/history", params={"limit": 3}).json():
            print("  ", h["step"], h["decision"], "|", h["headline"])