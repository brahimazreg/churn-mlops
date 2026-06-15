from joblib import dump
from src.data_processing import  load_data ,get_processor 
from src.config import *
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pathlib import Path
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("churn_prediction") 


MODELS = {
    "lr": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),

    "rfc": RandomForestClassifier(
        n_estimators=50,
        random_state=RANDOM_STATE
    ),

    "xgb": XGBClassifier(
        n_estimators=50,
        random_state=RANDOM_STATE
    )
}
#**************
def train_model(model):
    with mlflow.start_run(run_name=model.__class__.__name__):
        X, Y = load_data()
        preprocessor = get_processor()

        X_train, X_test, Y_train, Y_test = train_test_split(
            X,
            Y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=Y
        )

        # 🧪 TRAINING PIPELINE (WITH SMOTE)
        train_pipeline = ImbPipeline([
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model)
        ])

        train_pipeline.fit(X_train, Y_train)

        # 📊 evaluation
        accuracy, roc_auc, report = evaluate_model(
            train_pipeline, X_test, Y_test
        )
        
        inference_model = build_inference_pipeline(
            train_pipeline,
            preprocessor
        )

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_text(report, "report.txt")

        mlflow.sklearn.log_model(
            inference_model,
            artifact_path="model"
        )
        
        save_model(
            inference_model,
            model.__class__.__name__ 
        )

        return inference_model, X_test, Y_test
#*******************
def build_inference_pipeline(train_pipeline, preprocessor):

    model = train_pipeline.named_steps["model"]

    inference_pipeline = SkPipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])
    
    
    return inference_pipeline

def save_model(model, name):
    path = Path(MODELS_DIR) / f"{name}.joblib"
    dump(model, path)

if __name__ == "__main__":

    for model in MODELS.values():

        train_pipeline, X_test, Y_test = train_model(model)       

        

    
        
