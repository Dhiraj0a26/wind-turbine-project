/**
 * app.js — Wind Turbine Predictive Maintenance Frontend
 * Communicates with FastAPI backend at API_BASE.
 */

"use strict";

// ── Config ──────────────────────────────────────────────────
const API_BASE = "http://127.0.0.1:8000";

// ── Sensor definitions ──────────────────────────────────────
const SENSORS = [
  { key: "temperature",      label: "Temperature",      unit: "°C",   min: 20,   max: 130,  step: 0.5,  def: 62,   icon: "🌡️" },
  { key: "vibration",        label: "Vibration",        unit: "mm/s", min: 0.05, max: 20,   step: 0.05, def: 2.5,  icon: "📳" },
  { key: "pressure",         label: "Oil Pressure",     unit: "bar",  min: 0.5,  max: 7,    step: 0.05, def: 4.0,  icon: "💧" },
  { key: "rotational_speed", label: "Rotational Speed", unit: "RPM",  min: 400,  max: 2600, step: 10,   def: 1500, icon: "⚙️" },
  { key: "torque",           label: "Torque",           unit: "N·m",  min: 20,   max: 260,  step: 0.5,  def: 98,   icon: "🔩" },
  { key: "humidity",         label: "Humidity",         unit: "%",    min: 10,   max: 100,  step: 0.5,  def: 55,   icon: "💦" },
  { key: "oil_viscosity",    label: "Oil Viscosity",    unit: "cSt",  min: 15,   max: 90,   step: 0.5,  def: 62,   icon: "🛢️" },
  { key: "load_factor",      label: "Load Factor",      unit: "",     min: 0,    max: 1,    step: 0.01, def: 0.72, icon: "⚡" },
];

// Sensor display icons for state grid
const STATE_ICONS = {
  temperature:      "🌡️",
  vibration:        "📳",
  pressure:         "💧",
  rotational_speed: "⚙️",
  torque:           "🔩",
  humidity:         "💦",
  oil_viscosity:    "🛢️",
  load_factor:      "⚡",
};

// ── DOM refs ────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── State ───────────────────────────────────────────────────
let lastResult = null;

// ══════════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
  buildSliders();
  buildSensorGrid();
  checkApiHealth();
  loadModelInfo();

  $("predictBtn").addEventListener("click", runPrediction);
  $("modelSelect").addEventListener("change", () => {
    if (lastResult) runPrediction();   // re-run with new model
  });
});

// ══════════════════════════════════════════════════════════════
// BUILD SLIDERS
// ══════════════════════════════════════════════════════════════
function buildSliders() {
  const grid = $("slidersGrid");
  grid.innerHTML = "";

  SENSORS.forEach(s => {
    const wrap = document.createElement("div");
    wrap.className = "slider-item";
    wrap.innerHTML = `
      <div class="slider-header">
        <label class="slider-label">${s.icon} ${s.label}</label>
        <span class="slider-value" id="val_${s.key}">${s.def} ${s.unit}</span>
      </div>
      <input type="range"
             id="sl_${s.key}"
             min="${s.min}" max="${s.max}" step="${s.step}" value="${s.def}"
             oninput="onSlider('${s.key}','${s.unit}',this.value)" />
      <div class="slider-range">
        <span>${s.min} ${s.unit}</span>
        <span>${s.max} ${s.unit}</span>
      </div>`;
    grid.appendChild(wrap);
  });
}

function onSlider(key, unit, value) {
  $(`val_${key}`).textContent = `${parseFloat(value)} ${unit}`;
}

// ══════════════════════════════════════════════════════════════
// BUILD SENSOR STATE GRID (empty chips initially)
// ══════════════════════════════════════════════════════════════
function buildSensorGrid() {
  const grid = $("sensorGrid");
  grid.innerHTML = "";
  SENSORS.forEach(s => {
    const chip = document.createElement("div");
    chip.className  = "sensor-chip";
    chip.id         = `chip_${s.key}`;
    chip.innerHTML  = `
      <div class="sensor-chip-icon">${s.icon}</div>
      <div class="sensor-chip-name">${s.label}</div>
      <div class="sensor-chip-status" id="chipSt_${s.key}">—</div>`;
    grid.appendChild(chip);
  });
}

// ══════════════════════════════════════════════════════════════
// API HEALTH CHECK
// ══════════════════════════════════════════════════════════════
async function checkApiHealth() {
  const dot   = $("apiDot");
  const label = $("apiLabel");
  try {
    const res  = await fetch(`${API_BASE}/`);
    const data = await res.json();
    dot.className   = "api-dot online";
    label.textContent = `API Online · ${data.version}`;
  } catch {
    dot.className     = "api-dot offline";
    label.textContent = "API Offline";
  }
}

// ══════════════════════════════════════════════════════════════
// LOAD MODEL INFO & FEATURE IMPORTANCE
// ══════════════════════════════════════════════════════════════
async function loadModelInfo() {
  try {
    const [infoRes, fiRes] = await Promise.all([
      fetch(`${API_BASE}/api/model-info`),
      fetch(`${API_BASE}/api/feature-importance`),
    ]);
    if (infoRes.ok) renderComparisonTable(await infoRes.json());
    if (fiRes.ok)   renderFeatureChart(await fiRes.json());
  } catch (e) {
    console.warn("Could not load model info:", e);
  }
}

function renderComparisonTable(data) {
  const tbody   = $("comparisonBody");
  const metrics = data.model_comparison;
  if (!metrics || !Object.keys(metrics).length) return;

  // Find best by F1
  const bestModel = Object.entries(metrics)
    .reduce((best, [name, vals]) =>
      vals["F1-Score"] > (best[1]["F1-Score"] || 0) ? [name, vals] : best
    )[0];

  const rows = Object.entries(metrics).map(([name, vals]) => {
    const isBest = name === bestModel;
    const badge  = isBest ? `<span class="best-badge">Best</span>` : "";
    const fmtName = name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
    return `
      <tr class="${isBest ? "best-row" : ""}">
        <td>${fmtName}${badge}</td>
        <td>${pct(vals.Accuracy)}</td>
        <td>${pct(vals.Precision)}</td>
        <td>${pct(vals.Recall)}</td>
        <td>${pct(vals["F1-Score"])}</td>
        <td>${pct(vals["ROC-AUC"])}</td>
      </tr>`;
  });

  tbody.innerHTML = rows.join("");
}

function pct(v) {
  if (v == null) return "—";
  return `${(v * 100).toFixed(2)}%`;
}

function renderFeatureChart(data) {
  const container = $("featureChart");
  container.innerHTML = "";

  const entries = Object.entries(data.importances);      // sorted desc
  const maxVal  = entries[0][1];

  entries.forEach(([feat, imp]) => {
    const pctW = ((imp / maxVal) * 100).toFixed(1);
    const row  = document.createElement("div");
    row.className = "fi-bar-row fade-in";
    row.innerHTML = `
      <div class="fi-name">${feat}</div>
      <div class="fi-track">
        <div class="fi-fill" style="width:${pctW}%"></div>
      </div>
      <div class="fi-val">${(imp * 100).toFixed(2)}%</div>`;
    container.appendChild(row);
  });
}

// ══════════════════════════════════════════════════════════════
// PREDICTION
// ══════════════════════════════════════════════════════════════
async function runPrediction() {
  const btn = $("predictBtn");
  btn.classList.add("loading");
  btn.innerHTML = `<span class="btn-icon">⏳</span> Analysing…`;

  // Collect slider values
  const payload = { model_name: $("modelSelect").value };
  SENSORS.forEach(s => {
    payload[s.key] = parseFloat($(`sl_${s.key}`).value);
  });

  try {
    const res  = await fetch(`${API_BASE}/api/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    lastResult = data;
    renderResult(data);

  } catch (e) {
    showError(e.message);
  } finally {
    btn.classList.remove("loading");
    btn.innerHTML = `<span class="btn-icon">⚡</span> Analyse Machine State`;
  }
}

// ══════════════════════════════════════════════════════════════
// RENDER RESULT
// ══════════════════════════════════════════════════════════════
function renderResult(d) {
  const isFailure = d.prediction === 1;
  const prob      = d.failure_probability;
  const health    = d.health_score;

  // ── Banner ─────────────────────────────────────────────────
  const banner = $("statusBanner");
  banner.className = `status-banner ${isFailure ? "failure" : "normal"} fade-in`;
  $("bannerIcon").textContent  = isFailure ? "🔴" : "🟢";
  $("bannerTitle").textContent = isFailure ? "⚠️  GEARBOX FAILURE DETECTED" : "✅  NORMAL OPERATION";
  $("bannerSub").textContent   = isFailure
    ? `${(prob * 100).toFixed(1)}% failure probability — ${d.risk_level} Risk`
    : `${(prob * 100).toFixed(1)}% failure probability — System Healthy`;

  // ── KPIs ───────────────────────────────────────────────────
  const probKpi = $("kpiProb");
  probKpi.textContent = `${(prob * 100).toFixed(1)}%`;
  probKpi.style.color = riskColor(prob);

  const healthKpi = $("kpiHealth");
  healthKpi.textContent = `${health.toFixed(1)}`;
  healthKpi.style.color  = healthColor(health);

  const riskKpi = $("kpiRisk");
  riskKpi.textContent = d.risk_level;
  riskKpi.style.color  = riskColor(prob);

  $("kpiTime").textContent = `${d.processing_time_ms.toFixed(1)}ms`;

  // ── Gauge ──────────────────────────────────────────────────
  drawGauge(health);
  $("gaugeScore").textContent = health.toFixed(0);
  $("gaugeScore").style.color  = healthColor(health);

  // ── Probability bar ────────────────────────────────────────
  const pctStr = `${(prob * 100).toFixed(1)}%`;
  $("probFill").style.width  = pctStr;
  $("probThumb").style.left  = pctStr;

  // ── Sensor state chips ─────────────────────────────────────
  const state = d.machine_state;
  Object.entries(state).forEach(([key, status]) => {
    const chip = $(`chip_${key}`);
    const st   = $(`chipSt_${key}`);
    if (chip) {
      chip.className       = `sensor-chip ${status}`;
      st.textContent       = status.toUpperCase();
    }
  });

  // ── Alerts ─────────────────────────────────────────────────
  const alertsBox  = $("alertsBox");
  const alertsList = $("alertsList");
  if (d.alert_messages && d.alert_messages.length) {
    alertsList.innerHTML = d.alert_messages
      .map(a => `<div class="alert-item">${a}</div>`)
      .join("");
    alertsBox.style.display = "block";
    alertsBox.classList.add("fade-in");
  } else {
    alertsBox.style.display = "none";
  }

  // ── Recommendation ─────────────────────────────────────────
  const rec = $("recommendation");
  $("recText").textContent   = d.recommendation;
  rec.style.display          = "block";
  rec.style.borderLeftColor  = isFailure ? "var(--danger)" : "var(--success)";
  rec.classList.add("fade-in");

  // ── Footer ─────────────────────────────────────────────────
  const footer = $("resultFooter");
  $("footerModel").textContent    = d.model_used.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
  $("footerFeatures").textContent = `17`;
  $("footerTime").textContent     = `${d.processing_time_ms.toFixed(2)} ms`;
  footer.style.display            = "block";
}

// ══════════════════════════════════════════════════════════════
// GAUGE CANVAS
// ══════════════════════════════════════════════════════════════
function drawGauge(score) {
  const canvas = $("gaugeCanvas");
  const ctx    = canvas.getContext("2d");
  const cx = canvas.width / 2;
  const cy = canvas.height - 20;
  const r  = 110;
  const startAngle = Math.PI;
  const endAngle   = 2 * Math.PI;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Background arc
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, endAngle);
  ctx.lineWidth   = 18;
  ctx.strokeStyle = "#1E2D3D";
  ctx.lineCap     = "round";
  ctx.stroke();

  // Filled arc
  const fillEnd = startAngle + (score / 100) * Math.PI;
  const grad = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
  grad.addColorStop(0,   "#F85149");
  grad.addColorStop(0.5, "#D29922");
  grad.addColorStop(1,   "#3FB950");

  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, fillEnd);
  ctx.lineWidth   = 18;
  ctx.strokeStyle = grad;
  ctx.lineCap     = "round";
  ctx.stroke();

  // Tick marks
  for (let i = 0; i <= 10; i++) {
    const angle = Math.PI + (i / 10) * Math.PI;
    const ir = r - 24, or = r - 14;
    ctx.beginPath();
    ctx.moveTo(cx + ir * Math.cos(angle), cy + ir * Math.sin(angle));
    ctx.lineTo(cx + or * Math.cos(angle), cy + or * Math.sin(angle));
    ctx.lineWidth   = i % 5 === 0 ? 2 : 1;
    ctx.strokeStyle = "#3D5166";
    ctx.stroke();
  }
}

// ══════════════════════════════════════════════════════════════
// HELPERS
// ══════════════════════════════════════════════════════════════
function healthColor(h) {
  if (h >= 75) return "#3FB950";
  if (h >= 50) return "#D29922";
  return "#F85149";
}
function riskColor(prob) {
  if (prob < 0.25) return "#3FB950";
  if (prob < 0.50) return "#D29922";
  if (prob < 0.75) return "#F85149";
  return "#FF2222";
}

function showError(msg) {
  const banner = $("statusBanner");
  banner.className = "status-banner failure fade-in";
  $("bannerIcon").textContent  = "❌";
  $("bannerTitle").textContent = "Connection Error";
  $("bannerSub").textContent   = msg || "Could not reach API. Is the backend running?";
}
