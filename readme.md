# 📊 Churn Prediction MLOps Project

End-to-end Machine Learning + MLOps system for predicting customer churn.

---

## 🚀 Overview

This project includes:

- Data preprocessing & feature engineering
- ML models: Logistic Regression, Random Forest, XGBoost
- SMOTE for class imbalance
- MLflow experiment tracking
- FastAPI inference API
- Streamlit monitoring dashboard
- Docker containerization
- Pytest testing

---

## 🏗 Architecture

Raw Data → Processing → Train/Test Split → SMOTE → Model Training → Evaluation → MLflow Logging → FastAPI → Streamlit Dashboard

---

## 📁 Project Structure

- src/
- models/
- logs/
- tests/
- streamlit_app.py
- docker-compose.yml
- Dockerfile
- README.md

---

## ⚙️ Setup

### Clone repository

```bash
git clone https://github.com/brahimazreg/churn-mlops.git
cd churn-mlops

Install dependencies
--------------------

pip install -r requirements.txt

Run API (FastAPI)
Local  :  uvicorn src.api:app --reload
Docker :  docker compose up --build

API will be available at:
-------------------------

http://localhost:8000/docs

Streamlit Monitoring Dashboard
------------------------------
streamlit run streamlit_app.py

MLflow Tracking
---------------

mlflow ui 
then open : http://127.0.0.1:5000

Docker Services
----------------
FastAPI → http://localhost:8000
Swagger UI → http://localhost:8000/docs
Streamlit → http://localhost:8501

Testing
--------

pytest

Models Used
-----------
Logistic Regression
Random Forest
XGBoost

Best model: XGBoost (~0.82 ROC-AUC)

Example Prediction Request
--------------------------

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

Live Deployment
---------------
API:
https://churn-mlops-uh54.onrender.com

Swagger UI:
https://churn-mlops-uh54.onrender.com/docs


Technologies Used
-----------------
Python
Pandas
Scikit-learn
XGBoost
FastAPI
Streamlit
MLflow
Docker
Joblib

Author : Brahim Azreg

MLOps churn prediction project built for production-style machine learning practice.