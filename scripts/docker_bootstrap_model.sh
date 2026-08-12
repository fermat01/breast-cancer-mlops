#!/usr/bin/env bash

set -euo pipefail

echo "=========================================="
echo " Breast Cancer MLOps - Model Bootstrap"
echo "=========================================="

echo ""
echo "MLflow:"
echo "  $MLFLOW_TRACKING_URI"

echo ""
echo "Model:"
echo "  $MODEL_NAME"

echo ""
echo "Alias:"
echo "  $MODEL_ALIAS"

# ============================================================
# Wait for MLflow
# ============================================================

echo ""
echo "[1/4] Waiting for MLflow..."

python - <<'PY'
import os
import time
import urllib.request

url = os.environ["MLFLOW_TRACKING_URI"]

for attempt in range(30):
    try:
        urllib.request.urlopen(
            f"{url}/version",
            timeout=3,
        )
        print("MLflow is ready.")
        break
    except Exception:
        print(
            f"Waiting for MLflow... "
            f"attempt {attempt + 1}/30"
        )
        time.sleep(2)
else:
    raise RuntimeError("MLflow did not become ready.")
PY

# ============================================================
# Check champion alias
# ============================================================

echo ""
echo "[2/4] Checking model alias..."

if python - <<'PY'
import os
import sys

from mlflow.tracking import MlflowClient

client = MlflowClient(
    tracking_uri=os.environ["MLFLOW_TRACKING_URI"]
)

model_name = os.environ["MODEL_NAME"]
model_alias = os.environ["MODEL_ALIAS"]

try:
    version = client.get_model_version_by_alias(
        name=model_name,
        alias=model_alias,
    )

    print(
        f"Found model: "
        f"{model_name}@{model_alias} "
        f"(version={version.version})"
    )

    sys.exit(0)

except Exception:
    print(
        f"Model alias not found: "
        f"{model_name}@{model_alias}"
    )

    sys.exit(1)
PY
then

    echo ""
    echo "Champion model already exists."
    echo "Skipping training."

else

    # ========================================================
    # Train model
    # ========================================================

    echo ""
    echo "[3/4] Champion not found."
    echo "Training model..."

    python -m training.train

    echo ""
    echo "Training completed."

    # ========================================================
    # Promote model
    # ========================================================

    echo ""
    echo "[4/4] Assigning '$MODEL_ALIAS' alias..."

    python -m training.assign_model_alias

    echo ""
    echo "Champion model created successfully."

fi

echo ""
echo "=========================================="
echo " Model bootstrap completed"
echo "=========================================="