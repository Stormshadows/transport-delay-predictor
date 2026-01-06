import os
import json
import pandas as pd
import glob
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "raw")

files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

if not files:
    raise RuntimeError(f"No CSV files found in {DATA_DIR}")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df["start_time"] = pd.to_datetime(df["start_time"])
df = df[df["start_time"].dt.year >= 2018]

df["hour"] = df["start_time"].dt.hour
df["day_of_week"] = df["start_time"].dt.dayofweek
df["month"] = df["start_time"].dt.month
df["station_count"] = df["rdt_station_codes"].fillna("").apply(lambda x: len(x.split(",")) if x else 0)
df["line_count"] = df["rdt_lines_id"].fillna("").apply(lambda x: len(x.split(",")) if x else 0)

features = ["hour", "day_of_week", "month", "station_count", "line_count", "cause_group"]
target = "duration_minutes"

df = df[features + [target]].dropna()
df = pd.get_dummies(df, columns=["cause_group"], drop_first=True)

X = df.drop(columns=[target])
y = np.log1p(df[target])

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

FEATURE_COLUMNS = list(X.columns)

MODEL_PATH = os.path.join(BASE_DIR, "disruption_duration_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_columns.json")

with open(FEATURE_PATH, "w") as f:
    json.dump(FEATURE_COLUMNS, f)

model.fit(X, y)

joblib.dump(model, MODEL_PATH)

print("Model trained and saved")
print(f"Model path: {MODEL_PATH}")
print(f"Feature path: {FEATURE_PATH}")
