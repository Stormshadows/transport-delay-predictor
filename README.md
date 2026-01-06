# transport-delay-predictor


## Model Artifact
The trained disruption duration prediction model is stored in `models/disruption_duration_model.pkl` for reproducibility and ease of testing.


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
