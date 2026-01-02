import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def load_model(run_id):
    client = mlflow.tracking.MlflowClient()
    model_uri = f"runs:/{run_id}/iris_model"
    model = mlflow.sklearn.load_model(model_uri)
    return model

def predict(run_id, X):
    model = load_model(run_id)
    return model.predict(X)

if __name__ == "__main__":
    if os.path.exists("mlruns"):
        X, y = load_iris(return_X_y=True)
        runs = mlflow.search_runs(experiment_names=["iris-classification"])
        if not runs.empty:
            latest_run_id = runs.iloc[0]['run_id']
            predictions = predict(latest_run_id, X[:5])
            print(f"Predictions: {predictions}")
