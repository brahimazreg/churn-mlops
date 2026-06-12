from fastapi import FastAPI
from joblib import load
import pandas as pd
from src.predict import predict
from src.config import MODEL_PATH
from pydantic import BaseModel
from datetime import datetime
import json
from pathlib import Path
prediction_count = 0



app = FastAPI(title="Churn Prediction API",
              description="Predict customer churn using Logistic Regression",
              version="1.0.0"
              )

model = load(MODEL_PATH)

from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: int
    probability: float


LOG_FILE = Path("logs/predictions.jsonl")

def log_prediction(data, prediction, probability):
    LOG_FILE.parent.mkdir(exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": data,
        "prediction": prediction,
        "probability": probability
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict" , response_model=PredictionResponse)
def predict_api(data: CustomerInput):

    global prediction_count

    prediction_count += 1

    df = pd.DataFrame([data.model_dump()])
    pred, prob = predict(df)

    return {
        "prediction": int(pred[0]),
        "probability": float(prob[0])
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metrics")
def metrics():
    return {
        "total_predictions": prediction_count,
        "model_name": "LogisticRegression",
        "model_version": "1.0.0"
    }