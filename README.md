# transport-delay-predictor
A machine learning project that predicts the duration of major train disruptions in the Netherlands using historical NS (Nederlandse Spoorwegen) disruption data.

## Model Artifact
The trained disruption duration prediction model is stored in `models/disruption_duration_model.pkl` for reproducibility and ease of testing.

## 📌 Problem Statement

Train disruptions in the Netherlands can significantly impact commuters and logistics.  
This project aims to **predict the expected duration (in minutes)** of a major train disruption based on:

- Time of occurrence
- Number of affected stations and lines
- High-level cause category (e.g., Infrastructure, Weather, Staff)

Such predictions can support:
- Better passenger communication
- Operational planning
- Decision support systems

---

### Steps
- pip install -r requirements.txt
- python models/train_model.py
- uvicorn api.main:app --reload

### Example Request
```json
{
  "hour": 8,
  "day_of_week": 1,
  "month": 1,
  "station_count": 5,
  "line_count": 2,
  "cause_group": "Infrastructure"
}
```

### Example response
```json
{
  "predicted_duration_minutes": 75.34
}
```
