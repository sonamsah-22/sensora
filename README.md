# Sensora: AI-Based Sensor Trust, Drift and False-Alarm Detection

Sensora tells **genuine environmental events** apart from **faulty sensors**. It watches a stream of readings, decides whether an alarm should be trusted, scores how trustworthy each sensor is, and explains every decision in plain English.

Built by **Team EUREKA** 

> **The core question:** when a reading looks abnormal, is the environment dangerous, or is the sensor broken?
> A real event moves several related sensors together. A fault usually shows up on one sensor while its neighbours stay calm. Sensora uses that difference.

---

## What it does

|     Capability               |                               How                                                         |
|         ---                  |                               ---                                                         |
| Detects sensor faults        | Spike, stuck-at (frozen), excessive noise and gradual drift                               |
| Detects real events          | Smoke / fire, gas leak, poor air quality                                                  |
| Separates events from faults | Cross-sensor agreement, physical-consistency checks, and a fusion layer with safety rules |
| Scores sensor trust          | A 0 to 100 score per sensor that drops on fault evidence and recovers slowly              |
| Recommends maintenance       | OK, monitor, inspect, recalibrate or replace                                              |
| Reduces false alarms         | Genuine events are never dismissed silently; faults do not raise hazard alarms            |
| Explains decisions           | Plain-English evidence built from measurements an operator can verify                     |

**Sensors monitored (7): ** PM2.5, PM10, carbon monoxide, carbon dioxide, VOC, temperature, humidity.

---

## How it works

```
 raw readings
      |
      v
 rolling buffer (last 500 readings)
      |
      v
 feature engineering  -- per-sensor behaviour + cross-sensor agreement
      |
      +--> fault model (Random Forest)  : normal / spike / stuck / noise / drift, per sensor
      +--> event model (Random Forest)  : none / smoke_fire / gas_leak / poor_air_quality
      |
      v
 temporal smoothing (3 readings)
      |
      v
 fusion rules  -->  decision + alarm action
      |
      +--> trust tracker   (per-sensor trust 0-100 + maintenance status)
      +--> explanation     (evidence, reasoning, recommended actions)
      |
      v
 FastAPI  -->  React dashboard
```

### Decisions

|      Decision            |  Alarm action  |                    Meaning                              |
|       ---                |     ---        |                      ---                                |
| `NORMAL`                 | none           | Nothing wrong                                           |
| `GENUINE_EVENT`          | trigger alarm  | Real event, confirmed by several healthy sensors        |
| `GENUINE_EVENT_DEGRADED` | trigger alarm  | Real event, but one or more sensors in it are faulty    |
| `SENSOR_FAULT`           | suppress alarm | The anomaly is explained by a faulty sensor             |
| `NEEDS_VERIFICATION`     | verify         | Evidence is weak or not corroborated, so check manually |

### Safety rules in the fusion layer

- A possible hazard is never silently dismissed: if the event model fires and at least one healthy sensor agrees, the system asks for verification rather than suppressing the alarm.
- **Common-cause override:** if the event model is highly confident and 3 or more sensors move together, fault flags on those sensors are treated as unreliable. Several independent sensors failing in step with a known event pattern is far less likely than a false fault flag.

### Key features used by the models

|                           Feature                               |        Catches         |
|                             ---                                 |          ---           |
| `maxjump` (largest single-step change)                          | spikes                 |
| `zero_frac` (share of unchanged readings)                       | stuck sensors          |
| `dstd` (step-to-step jitter)                                    | noise                  |
| `zbase` (level vs a long-term baseline)                         | drift                  |
| `others_dev` and `n_dev` (how many other sensors deviate)       | event vs fault         |
| `resid_pm`, `resid_th` (PM10 vs PM2.5, humidity vs temperature) | physical inconsistency |

All features use only past readings, so the exact same code runs offline for training and live in the API.

---

## Project structure

```
sensora/
├── README.md
├── Dockerfile
├── requirements.txt
├── ml/                          # data + training
│   ├── common.py                # shared constants and paths
│   ├── generate_data.py         # synthetic stream with injected events and faults
│   ├── features.py              # window and cross-sensor features
│   ├── train_fault_model.py
│   └── train_event_model.py
├── models/                      # trained models (.joblib)
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py
│   │   ├── schemas.py
│   │   └── services/
│   │       ├── inference.py     # runs the two models
│   │       ├── fusion.py        # event-vs-fault decision rules
│   │       ├── trust.py         # trust score + maintenance status
│   │       ├── explain.py       # plain-English explanations
│   │       ├── monitor.py       # rolling buffer + full pipeline
│   │       └── simulator.py     # demo scenarios
│   └── tests/
└── frontend/                    # React dashboard (Vite)
    └── src/
        ├── main.jsx
        ├── App.jsx
        └── styles.css
```

---

## Getting started

**Requirements:** Python 3.12 and Node.js (LTS).

### 1. Install the backend

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate data and train the models

Run these from the project root, in order:

```bash
python ml/generate_data.py
python -m ml.train_fault_model
python -m ml.train_event_model
```

This writes `models/fault_model.joblib` and `models/event_model.joblib`. If you retrain, use the same library versions you deploy with, because saved models only load reliably with the scikit-learn version that created them.

### 3. Run the backend

```bash
uvicorn backend.app.main:app --reload
```

The interactive API docs are at `http://localhost:8000/docs`.

### 4. Run the dashboard (development)

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The dashboard calls the backend at `http://localhost:8000` (override with `VITE_API`).

### Production build (single server)

```bash
cd frontend && npm install && npm run build && cd ..
uvicorn backend.app.main:app
```

When `frontend/dist` exists, the backend serves the dashboard itself at `http://localhost:8000`.

---

## Using the dashboard

- **Scenarios:** pick one (normal, smoke / fire, gas leak, poor air quality, sensor spike, stuck sensor, noisy sensor, drifting sensor, or smoke plus a stuck sensor) and start the live simulation. One reading is analyzed every 2 seconds.
- **Live map:** sensors flow into the AI engine on the left, outcomes flow out on the right. Line thickness shows how far a sensor deviates from its recent baseline. Hover a sensor to trace it, click it for details and a "try a different value" box, or click an outcome to filter the log.
- **Replay:** drag the slider to review any earlier reading, then return to live.
- **Navigation:** the left sidebar jumps to the live map, what was measured, why the decision was made, what to do, and recent readings.

---

## API

| Method | Endpoint              |                           Purpose                                    |
|   ---  |         ---           |                             ---                                      |
| GET    | `/health`             | Service and model status                                             |
| POST   | `/analyze`            | Analyze one set of readings (all 7 sensors)                          |
| GET    | `/simulate/scenarios` | List demo scenarios                                                  |
| POST   | `/simulate/start`     | Start a scenario (`{"scenario": "smoke_fire"}`, optional `"sensor"`) |
| POST   | `/simulate/next`      | Produce and analyze the next simulated reading                       |
| GET    | `/latest`             | Most recent reading in the buffer                                    |
| GET    | `/trust`              | Current trust scores and maintenance status                          |
| GET    | `/history`            | Recent analyses                                                      |
| POST   | `/reset`              | Reset the monitor                                                    |

Example:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"readings": {"pm25": 12, "pm10": 23, "co": 0.4, "co2": 480, "voc": 90, "temp": 26, "humidity": 55}}'
```

Each response includes the decision, alarm action, per-sensor trust, evidence, reasoning and recommended actions. Simulation responses also include the ground truth.

---

## Testing

From the project root:

```bash
python -m backend.tests.test_fusion     # hand-built scenarios, including hard cases
python -m backend.tests.test_trust      # trust score and maintenance status
python -m backend.tests.test_explain    # prints readable explanations
pip install httpx
python -m backend.tests.test_api        # end-to-end, scored against ground truth
```

In our runs on synthetic scenarios, the fusion tests recorded **no hazards wrongly suppressed and no false alarms**, and in the end-to-end test the system raised **no false alarms on fault-only or all-clear steps**. Once a real event was established it kept the alarm raised on every step. Exact counts vary slightly between machines and runs.

---

## Deployment

The project ships as one container (React build plus FastAPI):

```bash
docker build -t sensora .
docker run -p 8000:8000 sensora
```

On a hosting service such as Render, create a Web Service from the repository using the **Docker** runtime. Commit the `models/` folder, which is small, and pin the library versions in `requirements.txt`. To host the frontend separately, set `VITE_API` to the backend URL at build time and add the frontend URL to the `CORS_ORIGINS` environment variable on the backend.

---

## Limitations

- **Synthetic data:** the models are trained and tested on data generated by `ml/generate_data.py`. Performance on real sensors is untested. Real-sensor integration is the main next step.
- **Drift is the hardest fault:** it develops slowly, so some drift episodes are only flagged late or as "needs verification".
- **Detection lag:** a few readings are needed before a new fault or event is recognised.
- **Single shared session:** the backend keeps one running simulation and buffer. Several visitors using one deployment at the same time will affect each other.
- **Thresholds are tunable:** trust penalties, the verification threshold and the fusion rules are constants in `trust.py` and `fusion.py`, chosen for this synthetic data.

## Future scope

- Integrate real sensors and real-time data feeds
- Expand datasets, including real fault logs
- Per-visitor sessions and persistent history
- Edge deployment on low-power devices
- Support for more sensor types

---

## Team

**Team EUREKA** 
1.Sonam Sah
2.Shivtej Nvale
3.Ritesh Bhakare
