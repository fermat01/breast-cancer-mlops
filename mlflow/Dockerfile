FROM ghcr.io/mlflow/mlflow:v3.0.0

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/

RUN uv pip install --system \
    psycopg2-binary \
    boto3