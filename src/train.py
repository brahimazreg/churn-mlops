

from joblib import dump

from src.data_processing import  load_data ,get_processor 
from src.config import *
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import mlflow


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("churn_prediction") 

print("TRACKING URI:", mlflow.get_tracking_uri())
print("EXPERIMENT:", mlflow.get_experiment_by_name("churn_prediction"))

def train_model():
    with mlflow.start_run(run_name="logistic_regression") as run:
        print("ACTIVE RUN:", mlflow.active_run())
        print("RUN ID:", run.info.run_id) 
        
        # call load_data function to get X , Y
        X,Y =load_data()

        # call get_processor function to get preprocessor
        preprocessor=get_processor()

        X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=Y
        )

        pipeline_lr = Pipeline([
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
        ])

        pipeline_lr.fit(X_train, Y_train)

        accuracy, roc_auc ,report = evaluate_model(
            pipeline_lr,
            X_test,
            Y_test
        )

        mlflow.log_param(
            "model",
            "LogisticRegression"
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE
        )

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )
        
      
        mlflow.log_text(
               report,
              "classification_report.txt"
        )
        mlflow.log_param("max_iter", 1000)

       

       

        mlflow.sklearn.log_model(
            sk_model=pipeline_lr,
            artifact_path="model"
        )

        save_model(pipeline_lr)
       

    return pipeline_lr,X_test,Y_test

def save_model(model):
    dump(model, MODEL_PATH) 



if __name__ == "__main__":
    train_model()


