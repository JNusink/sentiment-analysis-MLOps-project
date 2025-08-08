from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import json
from datetime import datetime
import os
from pathlib import Path

app = FastAPI()

# Load the pre-trained model
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Ensure logs directory exists
LOG_DIR = Path("/logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "prediction_logs.json"

# Request model
class PredictionRequest(BaseModel):
    text: str
    true_sentiment: str  # Expected to be 'positive' or 'negative'

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Predict sentiment
        prediction = model.predict([request.text])[0]
        predicted_sentiment = "positive" if prediction == 1 else "negative"

        # Create log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_text": request.text,
            "predicted_sentiment": predicted_sentiment,
            "true_sentiment": request.true_sentiment
        }

        # Append to log file
        with open(LOG_FILE, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")  # New line for each JSON object

        return {"predicted_sentiment": predicted_sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))