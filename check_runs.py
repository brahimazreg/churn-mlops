import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

exp = mlflow.get_experiment_by_name("churn_prediction")
print("Experiment:", exp)

runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id]
)

print("Number of runs:", len(runs))

if len(runs) > 0:
    print(runs[["run_id", "status"]])
else:
    print("No runs found")