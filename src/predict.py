from joblib import load
from src.config import MODEL_PATH
from src.features import add_features

model = load(MODEL_PATH)

def predict(df):

    df = add_features(df)

    pred = model.predict(df)
    prob = model.predict_proba(df)[:, 1]

    return pred, prob