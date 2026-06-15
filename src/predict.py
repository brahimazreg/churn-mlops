from joblib import load
from src.config import MODELS_DIR , DEFAULT_MODEL
from src.monitoring import log_prediction
from pathlib import Path 
import time

model_path = Path(MODELS_DIR) / DEFAULT_MODEL
model = load(model_path)


def predict(df):

    
    start = time.time()
    pred = model.predict(df)
    prob = model.predict_proba(df)[:, 1]

    print("PROB:", prob)   # TEMP DEBUG
    
    latency = time.time() - start
    # logging (safe conversion)
    log_prediction(
        df.to_dict(),
        int(pred[0]),
        float(prob[0]),
        latency
    )

    return pred, prob