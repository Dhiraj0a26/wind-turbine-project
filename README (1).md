# 🌬️ Wind Turbine Predictive Maintenance
## FastAPI Backend + HTML/JS Frontend

---

## 📁 Project Structure

```
wind_turbine_v2/
│
├── backend/
│   ├── main.py              ← FastAPI backend (all API endpoints)
│   ├── dataset.py           ← Dataset generator
│   ├── model.py             ← Model training pipeline
│   ├── requirements.txt     ← Python dependencies
│   ├── models/              ← Trained .pkl files (auto-generated)
│   └── data/                ← CSV dataset (auto-generated)
│
└── frontend/
    ├── index.html           ← Main UI page
    ├── css/
    │   └── style.css        ← All styling
    └── js/
        └── app.js           ← All JavaScript logic
```

---

## 🚀 How to Run

### Step 1 — Install Python dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 2 — Generate dataset + train models (if not done already)
```bash
python dataset.py
python model.py
```

### Step 3 — Start the FastAPI backend
```bash
uvicorn main:app --reload --port 8000
```
API is now live at: **http://127.0.0.1:8000**
Swagger docs at:   **http://127.0.0.1:8000/docs**

### Step 4 — Open the frontend
Simply open `frontend/index.html` in your browser.
*(No server needed — it's plain HTML/CSS/JS)*

> If you get a CORS error, serve the frontend with:
> ```bash
> cd frontend
> python -m http.server 5500
> ```
> Then open **http://localhost:5500**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/api/model-info` | Model metrics + dataset stats |
| GET | `/api/feature-importance` | RF feature importances |
| POST | `/api/predict` | Single sensor prediction |
| POST | `/api/predict-batch` | Batch predictions |

### Example POST /api/predict
```json
{
  "temperature": 88.5,
  "vibration": 7.2,
  "pressure": 2.3,
  "rotational_speed": 1480,
  "torque": 145,
  "humidity": 72,
  "oil_viscosity": 42,
  "load_factor": 0.61,
  "model_name": "random_forest"
}
```

### Example Response
```json
{
  "prediction": 1,
  "status": "FAILURE",
  "failure_probability": 0.9876,
  "health_score": 1.2,
  "risk_level": "Critical",
  "recommendation": "🚨 Immediate shutdown recommended.",
  "machine_state": {
    "temperature": "critical",
    "vibration": "critical",
    "pressure": "critical",
    ...
  },
  "alert_messages": ["⛔ Bearing overheating: 88.5°C", ...],
  "model_used": "random_forest",
  "processing_time_ms": 12.4
}
```

---

## 🎓 What to Explain to Teacher

- **dataset.py** → generates 30,000 realistic sensor rows
- **model.py** → trains 4 ML models, saves .pkl files
- **main.py** → FastAPI server, loads models, exposes REST API
- **index.html + app.js** → frontend calls the API via `fetch()`
- **Pydantic models** in `main.py` validate all input/output data
- **CORS middleware** allows the HTML page to call the API
