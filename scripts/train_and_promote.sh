#!/usr/bin/env bash

set -e

echo "=========================================="
echo " Breast Cancer MLOps - Train & Promote"
echo "=========================================="

# ============================================================
# 1. Configure environment for LOCAL training
# ============================================================

echo ""
echo "[1/5] Configuring MLflow and MinIO..."

export MLFLOW_TRACKING_URI="http://localhost:5000"

export MLFLOW_S3_ENDPOINT_URL="http://localhost:9010"

export AWS_ACCESS_KEY_ID="minio"

export AWS_SECRET_ACCESS_KEY="mlflow77"

export MODEL_NAME="breast-cancer-classifier"

export MODEL_ALIAS="champion"

echo "MLflow Tracking URI:"
echo "  $MLFLOW_TRACKING_URI"

echo "MinIO S3 Endpoint:"
echo "  $MLFLOW_S3_ENDPOINT_URL"

echo "Model:"
echo "  $MODEL_NAME"

echo "Alias:"
echo "  $MODEL_ALIAS"


# ============================================================
# 2. Check Docker services
# ============================================================

echo ""
echo "[2/5] Checking Docker services..."

docker compose ps


# ============================================================
# 3. Train model
# ============================================================

echo ""
echo "[3/5] Training model..."

python -m training.train

echo ""
echo "Training completed successfully."


# ============================================================
# 4. Assign champion alias
# ============================================================

echo ""
echo "[4/5] Assigning '$MODEL_ALIAS' alias..."

python -m training.assign_model_alias

echo ""
echo "Model alias assigned successfully."


# ============================================================
# 5. Finish
# ============================================================

echo ""
echo "[5/5] Deployment information"

echo "=========================================="
echo " Training pipeline completed"
echo "=========================================="

echo ""
echo "MLflow:"
echo "  http://localhost:5000"

echo ""
echo "MinIO:"
echo "  http://localhost:9010"

echo ""
echo "MinIO Console:"
echo "  http://localhost:9011"

echo ""
echo "FastAPI:"
echo "  http://localhost:8001"

echo ""
echo "Swagger:"
echo "  http://localhost:8001/docs"

echo ""
echo "Model:"
echo "  $MODEL_NAME"

echo ""
echo "Alias:"
echo "  $MODEL_ALIAS"

echo ""
echo "=========================================="