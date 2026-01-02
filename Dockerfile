FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
COPY app.py .
COPY main.py .
COPY predict.py .

RUN pip install --no-cache-dir mlflow scikit-learn numpy pandas flask

ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

EXPOSE 5000 8000

CMD ["python", "app.py"]
