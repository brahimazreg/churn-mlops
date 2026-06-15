from fastapi import FastAPI
from joblib import load
import pandas as pd
from src.predict import predict
from pydantic import BaseModel
from datetime import datetime
import json
from pathlib import Path
import time
from src.config import LOG_FILE
from src.config import MODELS_DIR , DEFAULT_MODEL


prediction_count = 0

app = FastAPI(title="Churn Prediction API",
              description="Predict customer churn using Logistic Regression",
              version="1.0.0"
              )

model_path = Path(MODELS_DIR) / DEFAULT_MODEL
model = load(model_path)

print("MODEL TYPE:", type(model)) # for debug



from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: int
    probability: float



# LOG_FILE = Path("../logs/predictions.jsonl")

def log_prediction(data, prediction, probability,latency):
    LOG_FILE.parent.mkdir(exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": data,
        "prediction": prediction,
        "probability": probability,
        "latency": latency 
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

    #global prediction_count
    start = time.time()
    #prediction_count += 1

    df = pd.DataFrame([data.model_dump()])
    pred, prob = predict(df)
    latency = time.time() - start
    
    log_prediction(
        data.model_dump(),
        int(pred[0]),
        float(prob[0]),
        latency
    )

    

    return {
        "prediction": int(pred[0]),
        "probability": float(prob[0]),
        "latency_ms": round(latency * 1000, 2)
    }



@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metrics")
def metrics():

    total = 0
    churn_1 = 0

    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as f:
            for line in f:
                record = json.loads(line)
                total += 1
                churn_1 += record["prediction"]

    return {
        "total_predictions": total,
        "churn_predictions": churn_1,
        "churn_rate": churn_1 / total if total > 0 else 0
    }