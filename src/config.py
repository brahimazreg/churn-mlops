# configuration variables
from pathlib import Path 

TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_PATH = "models/churn_model_lr.joblib"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "churn_prediction"

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "logs" / "predictions.jsonl"
