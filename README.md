# Sentiment Analysis MLOps Project

This project implements a multi-container MLOps application for sentiment analysis, consisting of a FastAPI prediction service and a Streamlit monitoring dashboard, communicating via a shared Docker volume. An evaluation script tests the API's performance using provided test data.

## System Architecture

- **FastAPI Prediction Service**: Serves sentiment predictions at `POST /predict`, logging requests to `/logs/prediction_logs.json`. Uses a fallback SVC model due to compatibility issues with the original `sentiment_model.pkl`.
- **Streamlit Monitoring Dashboard**: Visualizes data drift, target drift, and model performance (accuracy and precision) using `IMDB Dataset.csv` and logged predictions, with an alert if accuracy < 80%.
- **Docker Volume**: Persists and shares `prediction_logs.json` and `IMDB Dataset.csv` between containers.
- **Evaluation Script**: Tests the API using `test.json` and computes accuracy against the dataset.

## Prerequisites

- Docker
- Make
- Python 3.11 (for running evaluate.py locally, updated from 3.9+ due to container environment)
- `test.json`, `IMDB Dataset.csv`, and `sentiment_model.pkl` in the appropriate directories

## Setup and Running

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/JNusink/sentiment-analysis-mlops-project.git
   cd sentiment-analysis-mlops
   ```

2. **Ensure Files are Present**:
   - Place `test.json` in the root directory.
   - Place `IMDB Dataset.csv` in the `logs/` directory.
   - Place `sentiment_model.pkl` in the `api/` directory.

3. **Build and Run the Application**:
   ```bash
   make
   ```
   This builds and runs both containers. The FastAPI service runs on `http://localhost:8000`, and the Streamlit dashboard on `http://localhost:8501`.

4. **Access the Services**:
   - FastAPI: `http://localhost:8000/docs` for the interactive API documentation.
   - Streamlit: `http://localhost:8501` for the monitoring dashboard.

5. **Test the API with curl**:
   ```bash
   curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "This movie was great!", "true_sentiment": "positive"}'
   or invoke with PowerShell:
   Invoke-WebRequest -Uri "http://localhost:8000/predict" -Method Post -Headers @{ "Content-Type" = "application/json"; "accept" = "application/json" } -Body '{"text": "This movie was great!", "true_sentiment": "positive"}' -UseBasicParsing
   ```

6. **Run the Evaluation Script**:
   ```bash
   python evaluate.py
   ```
   Ensure the FastAPI service is running. The script sends requests to the API and prints the accuracy.

7. **Clean Up**:
   ```bash
   make clean
   ```
   Stops and removes containers, network, and volume.

## Project Structure

```
sentiment-analysis-mlops/
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py
│   └── sentiment_model.pkl
├── monitoring/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── dashboard.py
├── logs/
│   ├── IMDB Dataset.csv
│   └── prediction_logs.json (created at runtime)
├── test.json
├── evaluate.py
├── Makefile
└── README.md
```

## Notes

The FastAPI service logs each prediction with a timestamp, input text, predicted sentiment, and true sentiment to /logs/prediction_logs.json.
The Streamlit dashboard displays:

Data drift (sentence length distributions using IMDB Dataset.csv).
Target drift (sentiment distributions).
Model accuracy and precision, with an alert if accuracy < 80% (currently using fallback data).


The evaluation script uses test.json to test the API and compute accuracy.
Ensure Docker is running before executing make.
Ensure IMDB Dataset.csv has columns review and sentiment, renamed to text and true_label in the dashboard.