from joblib import load

model = load("models/XGBClassifier.joblib")
print(model)