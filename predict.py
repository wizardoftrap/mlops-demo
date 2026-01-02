import mlflow

def load_model(run_id):
    client = mlflow.tracking.MlflowClient()
    model_uri = f"runs:/{run_id}/iris_model"
    model = mlflow.sklearn.load_model(model_uri)
    return model

def predict(run_id, X):
    model = load_model(run_id)
    return model.predict(X)