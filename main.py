"""
==============================================================
  main.py  —  FastAPI Backend
  Project : Predictive Maintenance — Wind Turbine Gearboxes
==============================================================

Endpoints
---------
GET  /                    → Health check
GET  /api/model-info      → Model metadata & comparison table
POST /api/predict         → Single prediction from sensor input
POST /api/predict-batch   → Batch prediction from list of readings
GET  /api/feature-importance → RF feature importances

Run:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── paths ─────────────────────────────────────────────────────
BASE    = Path(__file__).parent
MODEL_DIR = BASE / "models"
DATA_PATH = BASE / "data" / "wind_turbine_sensor_data.csv"

# ── load artefacts once at startup ────────────────────────────
try:
    RF_MODEL      = joblib.load(MODEL_DIR / "random_forest.pkl")
    GB_MODEL      = joblib.load(MODEL_DIR / "gradient_boosting.pkl")
    DT_MODEL      = joblib.load(MODEL_DIR / "decision_tree.pkl")
    LR_MODEL      = joblib.load(MODEL_DIR / "logistic_regression.pkl")
    SCALER        = joblib.load(MODEL_DIR / "scaler.pkl")
    FEATURE_COLS  = joblib.load(MODEL_DIR / "feature_cols.pkl")
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    LOAD_ERROR    = str(e)

MODEL_MAP = {
    "random_forest":       RF_MODEL      if MODELS_LOADED else None,
    "gradient_boosting":   GB_MODEL      if MODELS_LOADED else None,
    "decision_tree":       DT_MODEL      if MODELS_LOADED else None,
    "logistic_regression": LR_MODEL      if MODELS_LOADED else None,
}

# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(
    title       = "Wind Turbine Predictive Maintenance API",
    description = "ML-powered gearbox fault detection using sensor fusion.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# Allow all origins so the plain HTML frontend can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ══════════════════════════════════════════════════════════════
#  PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════

class SensorInput(BaseModel):
    """Raw sensor readings sent from the frontend."""
    temperature:      float = Field(..., ge=20,  le=130,  description="Bearing temperature (°C)")
    vibration:        float = Field(..., ge=0.0, le=25.0, description="RMS vibration (mm/s)")
    pressure:         float = Field(..., ge=0.5, le=7.0,  description="Oil pressure (bar)")
    rotational_speed: float = Field(..., ge=400, le=2600, description="Shaft speed (RPM)")
    torque:           float = Field(..., ge=20,  le=260,  description="Drivetrain torque (N·m)")
    humidity:         float = Field(..., ge=10,  le=100,  description="Relative humidity (%)")
    oil_viscosity:    float = Field(..., ge=15,  le=90,   description="Oil viscosity (cSt)")
    load_factor:      float = Field(..., ge=0.0, le=1.0,  description="Electrical load factor")
    model_name:       str   = Field("random_forest",      description="ML model to use")

    @field_validator("model_name")
    @classmethod
    def check_model(cls, v: str) -> str:
        allowed = list(MODEL_MAP.keys())
        if v not in allowed:
            raise ValueError(f"model_name must be one of {allowed}")
        return v


class BatchSensorInput(BaseModel):
    readings: List[SensorInput]


class PredictionResponse(BaseModel):
    """Full prediction result returned to the frontend."""
    # Core result
    prediction:        int            # 0 = Normal, 1 = Failure
    status:            str            # "NORMAL" | "FAILURE"
    failure_probability: float        # 0.0 – 1.0
    health_score:      float          # 0 – 100
    risk_level:        str            # "Low" | "Medium" | "High" | "Critical"
    recommendation:    str

    # Machine state details
    machine_state:     dict           # Per-sensor status flags
    alert_messages:    List[str]      # Human-readable warnings

    # Meta
    model_used:        str
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    total:           int
    failure_count:   int
    normal_count:    int
    failure_rate:    float
    predictions:     List[PredictionResponse]


# ══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (mirrors dataset.py)
# ══════════════════════════════════════════════════════════════

def _engineer(raw: dict) -> dict:
    d = dict(raw)
    d["vibration_temp_ratio"]    = d["vibration"] / (d["temperature"] + 1e-9)
    d["pressure_speed_ratio"]    = d["pressure"]  / (d["rotational_speed"] + 1e-9)
    d["mechanical_stress_index"] = (d["torque"] * d["rotational_speed"]) / (d["pressure"] * 1000 + 1e-9)
    # Rolling proxies (single point → use raw value)
    for col in ["temperature", "vibration", "pressure"]:
        d[f"{col}_rolling_mean"] = d[col]
        d[f"{col}_rolling_std"]  = 0.0
    return d


def _build_row(sensor: SensorInput) -> pd.DataFrame:
    raw = sensor.model_dump(exclude={"model_name"})
    full = _engineer(raw)
    return pd.DataFrame([full])[FEATURE_COLS]


# ══════════════════════════════════════════════════════════════
#  MACHINE STATE ANALYSIS
# ══════════════════════════════════════════════════════════════

def _machine_state(s: SensorInput) -> tuple[dict, list[str]]:
    """
    Evaluate each sensor against safe operating thresholds.
    Returns (state_dict, alert_list).
    """
    state  = {}
    alerts = []

    checks = [
        ("temperature",      s.temperature,      80,   100,  "°C",  "Bearing overheating"),
        ("vibration",        s.vibration,         5,    10,  "mm/s","Excessive vibration"),
        ("pressure",         s.pressure,          None, None,"bar", None),          # low = bad
        ("rotational_speed", s.rotational_speed,  1800, 2200,"RPM", "Over-speed"),
        ("torque",           s.torque,            200,  240, "N·m", "High torque spike"),
        ("humidity",         s.humidity,          75,   90,  "%",   "High nacelle humidity"),
        ("oil_viscosity",    s.oil_viscosity,     None, None,"cSt", None),          # low = bad
        ("load_factor",      s.load_factor,       0.9,  0.98,"",    "Near full load"),
    ]

    for name, val, warn, crit, unit, msg in checks:
        # Special: low pressure is bad
        if name == "pressure":
            if val < 2.5:
                state[name] = "critical"
                alerts.append(f"⛔ Oil pressure critically low ({val:.2f} bar) — risk of lubrication failure")
            elif val < 3.0:
                state[name] = "warning"
                alerts.append(f"⚠️ Oil pressure low ({val:.2f} bar)")
            else:
                state[name] = "normal"
            continue

        # Special: low viscosity is bad
        if name == "oil_viscosity":
            if val < 30:
                state[name] = "critical"
                alerts.append(f"⛔ Oil viscosity very low ({val:.1f} cSt) — oil degradation suspected")
            elif val < 45:
                state[name] = "warning"
                alerts.append(f"⚠️ Oil viscosity below recommended range ({val:.1f} cSt)")
            else:
                state[name] = "normal"
            continue

        if val >= crit:
            state[name] = "critical"
            alerts.append(f"⛔ {msg}: {val:.1f} {unit}")
        elif val >= warn:
            state[name] = "warning"
            alerts.append(f"⚠️ {msg} approaching limit: {val:.1f} {unit}")
        else:
            state[name] = "normal"

    return state, alerts


def _risk_level(prob: float) -> str:
    if prob < 0.25:  return "Low"
    if prob < 0.50:  return "Medium"
    if prob < 0.75:  return "High"
    return "Critical"


def _recommendation(prob: float, prediction: int) -> str:
    if prediction == 0 and prob < 0.25:
        return "✅ No action required. Continue normal monitoring schedule."
    if prediction == 0 and prob < 0.50:
        return "🔍 Borderline reading. Increase sensor polling frequency."
    if prediction == 1 and prob < 0.75:
        return "⚠️ Schedule inspection within 48 hours. Check lubrication system."
    return "🚨 Immediate shutdown recommended. Dispatch maintenance team NOW."


# ══════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Wind Turbine Predictive Maintenance API",
        "status":  "online",
        "models_loaded": MODELS_LOADED,
        "version": "1.0.0",
    }


@app.get("/api/model-info", tags=["Info"])
def model_info():
    """Return model comparison metrics and dataset info."""
    comparison_path = MODEL_DIR / "model_comparison.csv"
    comparison = {}
    if comparison_path.exists():
        df = pd.read_csv(comparison_path, index_col=0)
        comparison = df.round(4).to_dict(orient="index")

    dataset_stats = {}
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        dataset_stats = {
            "total_rows":    len(df),
            "failure_count": int(df["failure"].sum()),
            "normal_count":  int((df["failure"] == 0).sum()),
            "failure_rate":  round(df["failure"].mean(), 4),
            "feature_count": len(df.columns) - 1,
        }

    return {
        "available_models": list(MODEL_MAP.keys()),
        "recommended_model": "random_forest",
        "feature_columns":  FEATURE_COLS,
        "model_comparison": comparison,
        "dataset_stats":    dataset_stats,
    }


@app.get("/api/feature-importance", tags=["Info"])
def feature_importance():
    """Return Random Forest feature importances sorted descending."""
    if not MODELS_LOADED:
        raise HTTPException(503, "Models not loaded")
    imp = dict(zip(FEATURE_COLS, RF_MODEL.feature_importances_.tolist()))
    imp_sorted = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
    return {"model": "random_forest", "importances": imp_sorted}


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(sensor: SensorInput):
    """
    Main prediction endpoint.
    Accepts sensor readings → returns full machine health report.
    """
    if not MODELS_LOADED:
        raise HTTPException(503, f"Models not loaded: {LOAD_ERROR}")

    t0  = time.perf_counter()
    clf = MODEL_MAP[sensor.model_name]
    row = _build_row(sensor)

    # Scale only for Logistic Regression
    row_input = SCALER.transform(row) if sensor.model_name == "logistic_regression" else row.values

    prob       = float(clf.predict_proba(row_input)[0, 1])
    prediction = int(prob >= 0.5)
    health     = round((1 - prob) * 100, 1)
    ms         = round((time.perf_counter() - t0) * 1000, 2)

    machine_state, alerts = _machine_state(sensor)

    return PredictionResponse(
        prediction           = prediction,
        status               = "FAILURE" if prediction else "NORMAL",
        failure_probability  = round(prob, 4),
        health_score         = health,
        risk_level           = _risk_level(prob),
        recommendation       = _recommendation(prob, prediction),
        machine_state        = machine_state,
        alert_messages       = alerts,
        model_used           = sensor.model_name,
        processing_time_ms   = ms,
    )


@app.post("/api/predict-batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(batch: BatchSensorInput):
    """Batch prediction for multiple sensor readings."""
    results = [predict(r) for r in batch.readings]
    failures = sum(r.prediction for r in results)
    return BatchPredictionResponse(
        total         = len(results),
        failure_count = failures,
        normal_count  = len(results) - failures,
        failure_rate  = round(failures / len(results), 4),
        predictions   = results,
    )
