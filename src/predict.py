from joblib import load
from src.config import MODEL_PATH
from src.features import add_features
from src.monitoring import log_prediction
import time


model = load(MODEL_PATH)

def predict(df):

    df = add_features(df)
    start = time.time()
    pred = model.predict(df)
    prob = model.predict_proba(df)[:, 1]
    
    latency = time.time() - start
    # logging (safe conversion)
    log_prediction(
        df.to_dict(),
        int(pred[0]),
        float(prob[0]),
        latency
    )

    return pred, prob