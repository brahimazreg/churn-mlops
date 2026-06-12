from src.predict import predict
import pandas as pd

def test_prediction():

    sample = pd.DataFrame({
        "gender": ["Male"],
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [12],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["Fiber optic"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["Yes"],
        "StreamingMovies": ["Yes"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [75.0],
        "TotalCharges": [900.0]
    })

    pred, prob = predict(sample)

    assert pred[0] in [0, 1]
    assert 0 <= prob[0] <= 1