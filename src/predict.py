from joblib import load
from src.config import MODEL_PATH

model = load(MODEL_PATH)


def predict(data):
    prediction = model.predict(data)
    probability = model.predict_proba(data)[:, 1]

    return prediction, probability