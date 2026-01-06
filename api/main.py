import os
import json
import joblib
import subprocess
import numpy as np
import pandas as pd
from fastapi import FastAPI

app = FastAPI(title="NS Disruption Duration Predictor")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "disruption_duration_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "..", "models", "feature_columns.json")

# Train model if missing
if not os.path.exists(MODEL_PATH):
    print("Model not found. Training model...")
    subprocess.run(["python", os.path.join(BASE_DIR, "..", "models", "train_model.py")], check=True)

# Load model
model = joblib.load(MODEL_PATH)
print("Model loaded successfully")

# Load feature columns
with open(FEATURE_PATH, "r") as f:
    FEATURE_COLUMNS = json.load(f)

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # Ensure all expected features exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df = df[FEATURE_COLUMNS]

    pred_log = model.predict(df)[0]
    prediction = float(np.expm1(pred_log))

    return {"predicted_duration_minutes": round(prediction, 2)}
