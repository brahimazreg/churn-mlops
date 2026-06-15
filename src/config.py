# configuration variables
from pathlib import Path 

TEST_SIZE = 0.2
RANDOM_STATE = 42
#DEFAULT_MODEL = "LogisticRegression.joblib"
#DEFAULT_MODEL = "RandomForestClassifier.joblib"
DEFAULT_MODEL = "XGBClassifier.joblib"
MODELS_DIR = "models"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "churn_prediction"

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "logs" / "predictions.jsonl"
