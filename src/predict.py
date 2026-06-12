from joblib import load
from src.config import MODEL_PATH
from src.features import add_features
from src.monitoring import log_prediction

model = load(MODEL_PATH)

def predict(df):

    df = add_features(df)

    pred = model.predict(df)
    prob = model.predict_proba(df)[:, 1]

    # logging (safe conversion)
    log_prediction(
        df.to_dict(),
        int(pred[0]),
        float(prob[0])
    )

    return pred, prob