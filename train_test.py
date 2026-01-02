import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-classification")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('f1_score', f1)
    
    mlflow.log_param('n_estimators', 100)
    mlflow.log_param('max_depth', 5)
    
    mlflow.sklearn.log_model(model, 'iris_model')
    
    print(f"accuracy={accuracy:.4f}")
    print(f"precision={precision:.4f}")
    print(f"recall={recall:.4f}")
    print(f"f1={f1:.4f}")
    
    MIN_ACCURACY = 0.8
    if accuracy < MIN_ACCURACY:
        print(f"FAILED: Accuracy {accuracy:.4f} < {MIN_ACCURACY}")
        sys.exit(1)
    
    print("PASSED: All metrics qualified")
    sys.exit(0)
