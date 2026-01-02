import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
from threading import Thread
import subprocess
import sys
import os
import time
from train import train_model

app = Flask(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = "sqlite:////app/mlflow.db"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("iris-classification")

def start_mlflow_server():
    subprocess.run([
        sys.executable, "-m", "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", "5000",
        "--backend-store-uri", MLFLOW_TRACKING_URI,
        "--default-artifact-root", "/app/mlruns"
    ])

def get_latest_model():
    try:
        runs = mlflow.search_runs(experiment_names=["iris-classification"])
        if not runs.empty:
            latest_run_id = runs.iloc[0]['run_id']
            model_uri = f"runs:/{latest_run_id}/iris_model"
            return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Error loading model: {e}")
    return None

@app.route('/train', methods=['POST'])
def train():
    try:
        train_model()
        return jsonify({"status": "success", "message": "Model trained successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features')
        
        if not features:
            return jsonify({"error": "No features provided"}), 400
        
        model = get_latest_model()
        if model is None:
            return jsonify({"error": "No trained model found. Train a model first using /train"}), 404
        
        prediction = model.predict([features])
        
        # Map to iris class names
        iris_classes = ['setosa', 'versicolor', 'virginica']
        predicted_class = iris_classes[int(prediction[0])]
        
        return jsonify({
            "prediction": int(prediction[0]),
            "class": predicted_class,
            "features": features
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    model = get_latest_model()
    return jsonify({
        "status": "healthy",
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "model_available": model is not None
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "service": "MLflow + Prediction Server",
        "endpoints": {
            "GET /": "This documentation",
            "GET /health": "Health check",
            "POST /train": "Train a new model",
            "POST /predict": "Make predictions with latest model"
        },
        "mlflow_ui": "http://localhost:5000"
    })

if __name__ == '__main__':
    # Start MLflow server in background thread
    print("Starting MLflow server...")
    mlflow_thread = Thread(target=start_mlflow_server, daemon=True)
    mlflow_thread.start()
    
    # Wait for MLflow to start
    time.sleep(3)
    
    # Train model on startup with latest code
    print("Training model with latest code...")
    train_model()
    
    print("\nMLflow Server on port 5000")
    print("Prediction API on port 8000")
    app.run(host='0.0.0.0', port=8000, debug=False)
