# Activate the virtual environment
source bcmlpos/bin/activate²

rm mlflow.db
rm -rf mlartifacts

python -m training.train


mlflow ui --backend-store-uri sqlite:///mlflow.db



#promoted with alias for Production

 python -m training.assign_model_alias

 

 # Test the API
uvicorn app.api.main:app --reload --port 8001

http://localhost:8001/health
http://127.0.0.1:8001/docs


curl http://localhost:8001/metrics



grafana password: bcmlpos



--------------------------------------------------------------

sqlite> SELECT
name,
version,
source,
run_id
FROM model_versions;

output 
breast-cancer-classifier|1|models:/m-b06e06fc0632495680297b29fea4454a|d625b7ec62594dc187381f6b9705bf01
which is :
Model characteristics:

name                    : breast-cancer-classifier
version                 : 1
source                  : models:/m-b06e06fc0632495680297b29fea4454a
run_id                  : d625b7ec62594dc187381f6b9705bf01  










But in a production AWS setup later, we will replace:

sqlite:///mlflow.db
file:///mlartifacts

with:

PostgreSQL/RDS
+
S3 artifact store


------------------------------------- Production --------------------------------------------------------

we can update it to use PostgreSQL as MLflow backend store and MinIO as S3-compatible artifact store. In fact, this is a better architecture for your MLOps 
project because it matches the production pattern you will later deploy on AWS





                 Training
                    |
                    |
              MLflow Tracking
                    |
        +-----------+------------+
        |                        |
        v                        v
 PostgreSQL                MinIO S3
(metadata)               (artifacts)
        |
        |
        v
 MLflow Model Registry
        |
        |
        v
 FastAPI
(load champion model)



------------------------- posgres database logging -----------------------------


docker compose exec postgres psql -U mlflow -d mlflow



SELECT
name,
version,
run_id
FROM model_versions;



-----------------------------------------------

export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=mlflow77
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9010
export MLFLOW_TRACKING_URI=http://localhost:5000

---- or 

set -a
source .env
set +a



-------------------------- shell scripts to verify the alias -----------------------------


python -c "
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri('http://localhost:5000')

client = MlflowClient()

model = client.get_model_version_by_alias(
    'breast-cancer-classifier',
    'champion'
)

print('Model:', model.name)
print('Version:', model.version)
print('Aliases:', model.aliases)
print('Source:', model.source)
"

# ----------------------------- From terminal get the features ---

python -c "from training.data_loader import load_dataset; d=load_dataset(); print(d.features.columns.tolist())"




# ------------------------------ Unit test the API -----------------------------

# predictor  unit test
pytest tests/unit/test_predictor.py -v
pytest -V


# Integration tests for the prediction API.
pytest tests/integration/test_prediction_api.py -v