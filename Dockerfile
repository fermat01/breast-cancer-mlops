FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir \
    -r requirements.txt

COPY app ./app
COPY training ./training
COPY scripts/docker_bootstrap_model.sh ./scripts/docker_bootstrap_model.sh

RUN chmod +x ./scripts/docker_bootstrap_model.sh

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]