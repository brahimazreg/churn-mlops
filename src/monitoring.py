from datetime import datetime
import json
from pathlib import Path

LOG_FILE = Path("logs/predictions.jsonl")

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "logs" / "predictions.jsonl"

LOG_FILE.parent.mkdir(exist_ok=True)

def log_prediction(input_data, prediction, probability, latency):

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "prediction": prediction,
        "probability": probability,
        "latency": latency
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")