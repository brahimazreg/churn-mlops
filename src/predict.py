from joblib import load
from src.config import MODEL_PATH
from src.features import add_features
from src.api import log_prediction


model = load(MODEL_PATH)

def predict(df):

    df = add_features(df)

    pred = model.predict(df)
    prob = model.predict_proba(df)[:, 1]

    result = {
        "prediction": int(pred[0]),
        "probability": float(prob[0])
        }

    log_prediction(
        df.model_dump(),
        result["prediction"],
        result["probability"]
)

    return result 