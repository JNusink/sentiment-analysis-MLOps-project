import streamlit as st
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score

# Load test data (IMDB Dataset.csv)
test_data = pd.read_csv("/logs/IMDB Dataset.csv")
test_data = test_data.rename(columns={"review": "text", "sentiment": "true_label"})

# Load prediction logs
LOG_FILE = Path("/logs/prediction_logs.json")
logs = []
if LOG_FILE.exists():
    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
logs_df = pd.DataFrame(logs)

# Streamlit app
st.title("Sentiment Model Monitoring Dashboard")

# Alert for low accuracy
if not logs_df.empty:
    true_labels = logs_df["true_sentiment"]
    predicted_labels = logs_df["predicted_sentiment"]
    accuracy = accuracy_score(true_labels, predicted_labels)
    if accuracy < 0.8:
        st.error("⚠️ Model accuracy is below 80%! Immediate attention required.")

# Data Drift Analysis
st.header("Data Drift Analysis")
st.write("Comparing sentence length distributions between training data and inference requests")
if not logs_df.empty:
    train_lengths = test_data["text"].str.len()
    inference_lengths = logs_df["request_text"].str.len()
    plt.figure(figsize=(10, 6))
    sns.kdeplot(train_lengths, label="Training Data", color="blue")
    sns.kdeplot(inference_lengths, label="Inference Requests", color="red")
    plt.xlabel("Sentence Length")
    plt.ylabel("Density")
    plt.legend()
    st.pyplot(plt)
else:
    st.write("No inference logs available for data drift analysis.")

# Target Drift Analysis
st.header("Target Drift Analysis")
st.write("Comparing sentiment distributions between training data and predictions")
if not logs_df.empty:
    train_sentiments = test_data["true_label"].value_counts(normalize=True)
    predicted_sentiments = logs_df["predicted_sentiment"].value_counts(normalize=True)
    comparison_df = pd.DataFrame({
        "Training Data": train_sentiments,
        "Predictions": predicted_sentiments
    }).fillna(0)
    st.bar_chart(comparison_df)
else:
    st.write("No inference logs available for target drift analysis.")

# Model Accuracy & Precision
st.header("Model Performance")
if not logs_df.empty:
    precision = precision_score(true_labels, predicted_labels, pos_label="positive")
    st.metric("Accuracy", f"{accuracy:.2%}")
    st.metric("Precision (Positive Class)", f"{precision:.2%}")
else:
    st.write("No feedback data available to compute performance metrics.")