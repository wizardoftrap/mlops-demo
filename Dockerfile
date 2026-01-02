FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
COPY main.py .

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -e .

ENV MLFLOW_TRACKING_URI=/app/mlruns

CMD ["python", "main.py"]
