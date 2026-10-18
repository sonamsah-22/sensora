import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API = import.meta.env.VITE_API ?? (import.meta.env.DEV ? "http://localhost:8000" : "");
const SENSORS = [["pm25", "PM2.5", "µg/m³"], ["pm10", "PM10", "µg/m³"], ["co", "Carbon monoxide", "ppm"],
  ["co2", "Carbon dioxide", "ppm"], ["voc", "VOC", "ppb"], ["temp", "Temperature", "°C"], ["humidity", "Humidity", "%"]];
const SCALE = { pm25: 3, pm10: 5, co: 0.1, co2: 40, voc: 20, temp: 1, humidity: 3 };
const DECISIONS = {
  NORMAL: { label: "Normal", tone: "ok" }, GENUINE_EVENT: { label: "Genuine event", tone: "event" },
  GENUINE_EVENT_DEGRADED: { label: "Genuine event, one sensor unreliable", tone: "event" },
  SENSOR_FAULT: { label: "Sensor fault", tone: "fault" }, NEEDS_VERIFICATION: { label: "Needs verification", tone: "verify" },
};
const OUT = [["TRIGGER_ALARM", "Alarm raised", "event", 70], ["VERIFY", "Needs verification", "verify", 160],
  ["SUPPRESS_ALARM", "Alarm suppressed", "fault", 280], ["NONE", "All clear", "ok", 370]];
const COLOR = { event: "#ff5d5d", verify: "#6db5ff", fault: "#f5b942", ok: "#2fd4bd" };
const KIND = { fault: "Faulty sensor", event: "Supports the event", possible_fault: "Possible fault" };
const ys = SENSORS.map((_, i) => 62 + i * 52);

async function call(path, body) {
  const res = await fetch(API + path, { method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body) });
  if (!res.ok) { const d = (await res.json().catch(() => ({}))).detail;
    throw new Error(typeof d === "string" ? d : "A value is outside the range that sensor can report."); }
  return res.json();
}
const clock = (iso) => new Date(iso).toLocaleTimeString();
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)] ?? 0; };
const partOfDay = (d) => { const h = d.getHours();
  if (h >= 5 && h < 12) return { key: "morning", text: "Good morning", icon: "sun" };
  if (h >= 12 && h < 17) return { key: "afternoon", text: "Good afternoon", icon: "sun" };
  if (h >= 17 && h < 21) return { key: "evening", text: "Good evening", icon: "sunset" };
  return { key: "night", text: "Working late", icon: "moon" }; };
const truthText = (t) => { if (!t) return "-"; const p = [];
  if (t.event !== "none") p.push(t.event.replaceAll("_", " ")); if (t.fault !== "none") p.push(`${t.fault} on ${t.faulty_sensor}`);
  return p.length ? p.join(" + ") : "all clear"; };

function Spark({ values }) {
  if (values.length < 2) return <svg className="spark" />;
  const lo = Math.min(...values), span = Math.max(...values) - lo || 1;
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * 100},${30 - ((v - lo) / span) * 26}`).join(" ");
  return <svg className="spark" viewBox="0 0 100 32" preserveAspectRatio="none"><polyline points={pts} /></svg>;
}

const ICONS = {
  map: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM12 3v3M12 18v3M3 12h3M18 12h3",
  measured: "M3 12h4l3-8 4 16 3-8h4",
  why: "M9 18h6M10 21h4M12 3a6 6 0 0 0-4 10.5c.7.7 1 1.5 1 2.5h6c0-1 .3-1.8 1-2.5A6 6 0 0 0 12 3z",
  todo: "M9 11l3 3 8-8M20 12v7a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h9",
  sun: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4",
  moon: "M21 13A9 9 0 1 1 11 3a7 7 0 0 0 10 10z",
  sunset: "M17 18a5 5 0 0 0-10 0M12 9V2M4.2 10.2l1.4 1.4M1 18h2M21 18h2M18.4 11.6l1.4-1.4M23 22H1M8 6l4-4 4 4",
  recent: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 7v5l3 2",
};
const Icon = ({ n }) => <svg className="ic" viewBox="0 0 24 24" aria-hidden="true"><path d={ICONS[n]} /></svg>;
const NAV = [["sec-map", "map", "Live map"], ["sec-measured", "measured", "Measured"], ["sec-why", "why", "Why"],
  ["sec-todo", "todo", "What to do"], ["sec-recent", "recent", "Recent readings"]];

const RINGS = [[44, 24, ""], [62, 36, "b"], [80, 48, "c"]].map(([r, n, c]) => ({ c,
  dots: Array.from({ length: n }, (_, i) => { const a = (i / n) * 2 * Math.PI; return [Math.cos(a) * r, Math.sin(a) * r, 1 + ((i * 7) % 3) * 0.7]; }) }));

function Hero({ r, log, hover, setHover, open, filter, setFilter }) {
  const total = Math.max(log.length, 1);
  const tone = r ? DECISIONS[r.decision].tone : "idle";
  const recent = log.slice(-60);
  const dev = (k) => { if (recent.length < 6) return 0;
    const v = recent.map((x) => x.readings[k]); return Math.min(Math.abs(v[v.length - 1] - median(v)) / SCALE[k] / 3, 1); };
  return (
    <div className={`hero ${tone}`}>
      <svg className="flowmap" viewBox="0 0 1000 440" role="img" aria-label="Sensor data flowing through the AI engine to alarm outcomes">
        <text className="cap" x="170" y="22">Sensors</text><text className="cap" x="830" y="22">Outcomes</text>
        {SENSORS.map(([k, name, unit], i) => {
          const t = r?.trust.sensors[k]; const c = r?.faulty_sensors[k] ? "fault" : r?.corroborating_sensors.includes(k) ? "event" : "";
          const d = `M170,${ys[i]} C320,${ys[i]} 300,220 420,220`;
          return (
            <g key={k} className={`src ${hover === k ? "on" : hover ? "dim" : ""}`} tabIndex={0} role="button"
              aria-label={`${name}: open details`} onMouseEnter={() => setHover(k)} onMouseLeave={() => setHover(null)}
              onFocus={() => setHover(k)} onBlur={() => setHover(null)} onClick={() => open(k)}
              onKeyDown={(e) => e.key === "Enter" && open(k)}>
              <path className={`flow ${c}`} d={d} strokeWidth={1.5 + dev(k) * 9} />
              <path className="dash" d={d} />
              <circle className="node" cx="170" cy={ys[i]} r="7" fill={!t ? "#3a465c" : t.status === "OK" ? "#2fd4bd" : t.status === "MONITOR" ? "#6db5ff" : "#f5b942"} />
              <text className="lbl" x="154" y={ys[i] + 4} textAnchor="end">{name}</text>
              {r && <text className="val" x="154" y={ys[i] + 18} textAnchor="end">{r.readings[k].toFixed(1)} {unit}</text>}
              <rect x="0" y={ys[i] - 22} width="440" height="44" fill="transparent" />
            </g>);
        })}
        <g transform="translate(500,220)">
          {RINGS.map((ring, i) => <g key={i} className={`ring ${ring.c}`}>{ring.dots.map(([x, y, s], j) => <circle key={j} className="dot" cx={x} cy={y} r={s} opacity={0.35 + ((i + j) % 4) * 0.2} />)}</g>)}
          <text className="big" y="8">{r?.step ?? 0}</text><text className="cap" y="28" textAnchor="middle">readings</text>
        </g>
        {OUT.map(([key, label, t, oy]) => {
          const n = log.filter((x) => x.alarm_action === key).length; const live = r?.alarm_action === key;
          const d = `M582,220 C700,220 720,${oy} 826,${oy}`;
          return (
            <g key={key} className={`out ${live ? "live" : ""} ${filter === key ? "sel" : ""}`} style={{ "--c": COLOR[t] }} tabIndex={0} role="button"
              aria-pressed={filter === key} aria-label={`${label}: ${n} readings. Filter history`}
              onClick={() => setFilter(filter === key ? null : key)} onKeyDown={(e) => e.key === "Enter" && setFilter(filter === key ? null : key)}>
              <path className="flow" d={d} strokeWidth={2 + (n / total) * 22} />{live && <path className="dash" d={d} />}
              <circle className="node" cx="836" cy={oy} r="9" fill={COLOR[t]} />
              <text className="lbl" x="854" y={oy - 2}>{label}</text><text className="val" x="854" y={oy + 14}>{n} reading{n === 1 ? "" : "s"}</text>
              <rect x="700" y={oy - 24} width="300" height="48" fill="transparent" />
            </g>);
        })}
      </svg>
      {hover && r && (() => { const t = r.trust.sensors[hover]; const [, name, unit] = SENSORS.find((s) => s[0] === hover);
        return <div className="tip"><span>{name}</span><b>{r.readings[hover].toFixed(2)} {unit}</b>Trust {Math.round(t.trust)}% · {t.status.toLowerCase()}<br />Click for details</div>; })()}
    </div>
  );
}

function Drawer({ k, r, log, close, tryValue }) {
  const [name, unit] = SENSORS.find((s) => s[0] === k).slice(1); const t = r.trust.sensors[k];
  const [v, setV] = useState(r.readings[k].toFixed(2));
  useEffect(() => { const f = (e) => e.key === "Escape" && close(); window.addEventListener("keydown", f); return () => window.removeEventListener("keydown", f); }, [close]);
  return (
    <aside className="drawer" aria-label={`${name} details`}>
      <button className="pill x" onClick={close}>Close</button>
      <h2>{name}</h2><p className="sub">{r.readings[k].toFixed(2)} {unit} · {t.status.toLowerCase()}</p>
      <h3 style={{ marginTop: 18 }}>Trust {Math.round(t.trust)}%</h3>
      <div className="bartrack"><i style={{ width: `${t.trust}%`, background: t.trust >= 70 ? "var(--ok)" : t.trust >= 40 ? "var(--fault)" : "var(--event)" }} /></div>
      <Spark values={log.slice(-60).map((x) => x.readings[k])} />
      <h3 style={{ marginTop: 18 }}>Maintenance advice</h3>
      <p style={{ margin: 0 }}>{t.action}</p><p className="sub">{t.reason}</p>
      {r.explanation.evidence.filter((e) => e.sensor === k).map((e, i) => <p key={i}>{e.text}</p>)}
      <h3 style={{ marginTop: 18 }}>Try a different value</h3>
      <p className="sub">Changes only this sensor and re-runs the analysis.</p>
      <input aria-label={`New ${name} value`} inputMode="decimal" value={v} onChange={(e) => setV(e.target.value)} />
      <button className="pill go" onClick={() => tryValue(k, v)}>Analyze this value</button>
    </aside>
  );
}

export default function App() {
  const [scenarios, setScenarios] = useState([]); const [scenario, setScenario] = useState("smoke_fire");
  const [sensor, setSensor] = useState(""); const [running, setRunning] = useState(false);
  const [log, setLog] = useState([]); const [view, setView] = useState(null);
  const [hover, setHover] = useState(null); const [sel, setSel] = useState(null); const [filter, setFilter] = useState(null);
  const [error, setError] = useState(""); const busy = useRef(false); const started = useRef(false);

  const r = log.length ? log[view ?? log.length - 1] : null;
  const show = useCallback((x) => { setLog((l) => [...l.slice(-199), x]); setView(null); setError(""); }, []);
  useEffect(() => { call("/simulate/scenarios").then(setScenarios)
    .catch(() => setError(`Can't reach the backend at ${API || "this server"}. Start it with: uvicorn backend.app.main:app --reload`)); }, []);
  const begin = async () => { const needs = scenarios.find((s) => s.id === scenario)?.needs_sensor;
    await call("/simulate/start", { scenario, sensor: needs && sensor ? sensor : null }); started.current = true; setLog([]); setFilter(null); };
  const tick = useCallback(async () => { if (busy.current) return; busy.current = true;
    try { show(await call("/simulate/next", {})); } catch (e) { setError(e.message); setRunning(false); } finally { busy.current = false; } }, [show]);
  useEffect(() => { if (!running) return; const id = setInterval(tick, 2000); return () => clearInterval(id); }, [running, tick]);
  const toggle = async () => { if (running) return setRunning(false); try { await begin(); setRunning(true); tick(); } catch (e) { setError(e.message); } };
  const stepOnce = async () => { try { if (!started.current) await begin(); await tick(); } catch (e) { setError(e.message); } };
  const tryValue = async (k, v) => { try { const n = parseFloat(v); if (Number.isNaN(n)) throw new Error("Enter a number.");
    show(await call("/analyze", { readings: { ...r.readings, [k]: n } })); } catch (e) { setError(e.message); } };
  const pick = (set) => (e) => { set(e.target.value); started.current = false; };
  const [now, setNow] = useState(() => new Date());
  useEffect(() => { const id = setInterval(() => setNow(new Date()), 30000); return () => clearInterval(id); }, []);
  const pod = partOfDay(now);
  const has = r !== null;
  const [active, setActive] = useState("sec-map");
  useEffect(() => {
    const f = () => { let cur = "sec-map";
      for (const [id] of NAV) { const el = document.getElementById(id); if (el && el.getBoundingClientRect().top < 140) cur = id; }
      if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 4 && document.getElementById("sec-recent")) cur = "sec-recent";
      setActive(cur); };
    f(); window.addEventListener("scroll", f, { passive: true }); return () => window.removeEventListener("scroll", f);
  }, [has]);
  const go = (id) => document.getElementById(id)?.scrollIntoView({
    behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  const needsSensor = scenarios.find((s) => s.id === scenario)?.needs_sensor;

  const stats = useMemo(() => ({ trust: log.map((x) => x.trust.average_trust),
    raised: log.filter((x) => x.alarm_action === "TRIGGER_ALARM").length, prevented: log.filter((x) => x.alarm_action === "SUPPRESS_ALARM").length }), [log]);
  const rows = (filter ? log.filter((x) => x.alarm_action === filter) : log).slice(-8).reverse();
  const d = r ? DECISIONS[r.decision] : null;

  return (
    <div className="layout" data-time={pod.key}>
      <aside className="side">
        <div className="brandv">Sensora<small>Sensor intelligence</small></div>
        <nav aria-label="Sections">
          {NAV.map(([id, ic, label]) => (
            <button key={id} className="nv" disabled={id !== "sec-map" && !has}
              aria-current={active === id ? "true" : undefined} onClick={() => go(id)}>
              <Icon n={ic} />{label}{id === "sec-recent" && <span className="badge">{log.length}</span>}
            </button>))}
        </nav>
      </aside>
    <div className="page">
      <div className="bar">
        <span className="brand"><i className={`live ${running ? "on" : ""}`} />Command center<em>{running ? "Simulation live" : log.length ? "Paused" : "Idle"}</em></span>
        <div className="ctl">
          <select aria-label="Scenario" value={scenario} onChange={pick(setScenario)} disabled={running}>
            {scenarios.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}</select>
          {needsSensor && <select aria-label="Sensor to break" value={sensor} onChange={pick(setSensor)} disabled={running}>
            <option value="">Random sensor</option>{SENSORS.map(([k, n]) => <option key={k} value={k}>{n}</option>)}</select>}
          <button className="pill go" onClick={toggle} disabled={!scenarios.length}>{running ? "Pause" : "Start live simulation"}</button>
          <button className="pill" onClick={stepOnce} disabled={running || !scenarios.length}>Next reading</button>
        </div>
      </div>
      <div className="greet"><span className="tod"><Icon n={pod.icon} /></span>
        <div><h1>{pod.text}, operator</h1>
          <p className="when">{now.toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" })} at {now.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}</p></div></div>
      <p className="sub">{r ? "Hover a sensor to trace its signal. Click one for details, or click an outcome to filter the log." : "Pick a scenario and start the simulation to watch the system decide."}</p>
      {error && <div className="err" role="alert">{error}</div>}
      <div id="sec-map" />
      <div className="heroframe">
        <Hero r={r} log={log} hover={hover} setHover={setHover} open={setSel} filter={filter} setFilter={setFilter} />
        <div className="legend" aria-hidden="true">
          <span><i style={{ background: "#2fd4bd" }} />Healthy signal</span><span><i style={{ background: "#f5b942" }} />Faulty sensor</span>
          <span><i style={{ background: "#ff5d5d" }} />Supports a real event</span><span><i style={{ background: "#6db5ff" }} />Needs verification</span>
        </div>
      </div>
      {log.length > 1 && <div className="replay"><span>Replay</span>
        <input type="range" aria-label="Replay earlier readings" min="0" max={log.length - 1} value={view ?? log.length - 1} onChange={(e) => setView(+e.target.value === log.length - 1 ? null : +e.target.value)} />
        <button className="pill" onClick={() => setView(null)} disabled={view === null}>{view === null ? "Live" : "Back to live"}</button></div>}
      <div className="stats">
        <div className="stat"><small>Average sensor trust</small><b>{r ? `${r.trust.average_trust}%` : "-"}</b><Spark values={stats.trust.slice(-60)} /></div>
        <div className="stat"><small>Readings analyzed</small><b>{log.length}</b></div>
        <div className="stat"><small>Alarms raised</small><b>{stats.raised}</b></div>
        <div className="stat"><small>False alarms prevented</small><b>{stats.prevented}</b></div>
      </div>
      {r && <>
        <div className={`verdict ${d.tone}`}><span>{d.label}</span><h2>{r.explanation.headline}</h2><p>{r.explanation.summary}</p></div>
        <div className="cards3">
          <section id="sec-measured" className="xcard"><h3><Icon n="measured" />What the system measured</h3>
            {r.explanation.evidence.length === 0 ? <p className="sub">Nothing unusual in the recent readings.</p> :
              <ul className="list">{r.explanation.evidence.map((x, i) => <li key={i} className={x.kind}><small>{KIND[x.kind]}</small>{x.text}</li>)}</ul>}</section>
          <section id="sec-why" className="xcard"><h3><Icon n="why" />Why this decision</h3>
            <ul className="list">{r.explanation.why.map((w, i) => <li key={i}>{w}</li>)}</ul></section>
          <section id="sec-todo" className="xcard"><h3><Icon n="todo" />What to do</h3>
            <ul className="list">{r.explanation.recommended_actions.map((x, i) => <li key={i}>{x}</li>)}</ul></section>
        </div>
        <section id="sec-recent" className="xcard wide">
          <h3><Icon n="recent" />Recent readings{filter ? ` · ${OUT.find((o) => o[0] === filter)[1]}` : ""}
            {filter && <button className="pill" style={{ marginLeft: 8 }} onClick={() => setFilter(null)}>Clear filter</button>}</h3>
          <div className="tablewrap"><table><thead><tr><th>Reading</th><th>Time</th><th>Decision</th><th>Summary</th><th>Simulation truth</th></tr></thead>
            <tbody>{rows.map((x) => <tr key={x.step}><td>{x.step}</td><td>{clock(x.timestamp)}</td>
              <td><span className={`chip ${DECISIONS[x.decision].tone}`}>{DECISIONS[x.decision].label}</span></td><td>{x.explanation.headline}</td><td>{truthText(x.truth)}</td></tr>)}</tbody></table></div>
        </section>
      </>}
      {sel && r && <Drawer key={sel} k={sel} r={r} log={log} close={() => setSel(null)} tryValue={tryValue} />}
    </div>
    </div>
  );
}