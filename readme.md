# Churn Prediction MLOps Project

## Overview

This project implements an end-to-end MLOps pipeline for customer churn prediction using a Logistic Regression model. It includes:

* Model training and evaluation
* Experiment tracking with MLflow
* FastAPI prediction service
* Prediction logging
* Streamlit monitoring dashboard
* Basic model monitoring metrics

## Project Structure

```text
churn-mlops/
│
├── src/
│   ├── api.py
│   ├── predict.py
│   ├── train.py
│   ├── config.py
│   └── ...
│
├── models/
│   └── churn_model_lr.joblib
│
├── logs/
│   └── predictions.jsonl
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── mlflow.db
```

---

## Features

### Model Training

* Logistic Regression classifier
* Data preprocessing and feature engineering
* Model serialization with Joblib

### Experiment Tracking

* MLflow integration
* Parameter logging
* Metrics logging
* Model version tracking

### Prediction API

Built with FastAPI.

Endpoints:

| Method | Endpoint   | Description            |
| ------ | ---------- | ---------------------- |
| GET    | `/`        | Health message         |
| GET    | `/health`  | Service health check   |
| POST   | `/predict` | Predict customer churn |
| GET    | `/metrics` | Monitoring metrics     |

### Monitoring Dashboard

Built with Streamlit.

Dashboard displays:

* Total predictions
* Churn rate
* Average latency
* Prediction distribution
* Latency trends

---

## Installation

### Clone Repository

```bash
git clone https://github.com/brahimazreg/churn-mlops.git
cd churn-mlops
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the API

Start the FastAPI server:

```bash
uvicorn src.api:app --reload
```

API documentation:

```text
http://localhost:8000/docs
```

---

## Running the Dashboard

Start Streamlit:

```bash
streamlit run streamlit_app.py
```

Dashboard URL:

```text
http://localhost:8501
```

---

## Example Prediction Request

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 75.0,
  "TotalCharges": 900.0
}
```

Response:

```json
{
  "prediction": 1,
  "probability": 0.90,
  "latency_ms": 42.15
}
```

---

## Logging

Predictions are stored in:

```text
logs/predictions.jsonl
```

Each prediction record contains:

* Timestamp
* Input features
* Prediction
* Probability
* Inference latency

---

## Monitoring Metrics

The API exposes monitoring statistics through:
https://churn-mlops-uh54.onrender.com 
```text
GET /metrics
```

Example response:

```json
{
  "total_predictions": 25,
  "churn_predictions": 18,
  "churn_rate": 0.72
}
```

---

## Future Improvements

* Model drift detection
* Data quality monitoring
* Prometheus integration
* Grafana dashboards
* Docker containerization
* CI/CD pipeline with GitHub Actions
* Automated retraining

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* FastAPI
* Streamlit
* MLflow
* Joblib
* Uvicorn

---

## Author   

Brahim Azreg

Developed as an MLOps learning project demonstrating model deployment, monitoring, and experiment tracking workflows.
