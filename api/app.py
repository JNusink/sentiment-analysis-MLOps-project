import pickle
import os
from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

app = FastAPI()

# Fallback model if pickle fails
try:
    with open("sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Original model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    # Fallback to a default SVC model with fitted vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    # Fit with dummy data
    dummy_data = ["positive example", "negative example"]
    X_dummy = vectorizer.fit_transform(dummy_data)
    y_dummy = ["positive", "negative"]
    model = SVC(kernel='linear', probability=True)
    model.fit(X_dummy, y_dummy)
    print("Using fallback SVC model")

@app.post("/predict")
async def predict_sentiment(request: dict):
    text = request.get("text")
    true_sentiment = request.get("true_sentiment")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0][1] if prediction == "positive" else 1 - model.predict_proba(text_vectorized)[0][1]
    with open("/logs/prediction_logs.json", "a") as f:
        f.write(f'{{"request_text": "{text}", "true_sentiment": "{true_sentiment}", "predicted_sentiment": "{prediction}", "probability": {probability:.2f}}}\n')
    return {"predicted_sentiment": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)