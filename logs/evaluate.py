import json
import requests
from sklearn.metrics import accuracy_score

# Load test data
with open("test.json", "r") as f:
    test_data = json.load(f)

# Initialize lists for predictions and true labels
predictions = []
true_labels = []

# Send requests to FastAPI
url = "http://localhost:8000/predict"
for item in test_data:
    payload = {
        "text": item["text"],
        "true_sentiment": item["true_label"]
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        predicted_sentiment = response.json()["predicted_sentiment"]
        predictions.append(predicted_sentiment)
        true_labels.append(item["true_label"])
    except requests.RequestException as e:
        print(f"Error processing text: {item['text'][:50]}... Error: {e}")

# Calculate and print accuracy
if predictions:
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Model Accuracy: {accuracy:.2%}")
else:
    print("No successful predictions were made.")