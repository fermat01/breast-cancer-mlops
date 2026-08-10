# Activate the virtual environment
source bcmlpos/bin/activate²

rm mlflow.db
rm -rf mlartifacts

python -m training.train

#promoted with alias for Production

 python -m training.assign_model_alias


mlflow ui --backend-store-uri sqlite:///mlflow.db

#####################

## Start docker compose with env for prod

docker exec breast-cancer-api env | grep -E 'MLFLOW|AWS|MODEL'

docker compose --env-file .env config

docker compose --env-file .env up -d 

docker compose --env-file .env down



                  HOST MACHINE
                       │
          ┌────────────┴────────────┐
          │                         │
   localhost:8001             localhost:5000
          │                         │
          ▼                         ▼
    ┌───────────┐             ┌───────────┐
    │  FastAPI  │             │  MLflow   │
    │ container │             │ container │
    └─────┬─────┘             └─────┬─────┘
          │                         │
          │    Docker network       │
          │                         │
          └──────────┬──────────────┘
                     │
                     ▼
                ┌─────────┐
                │  MinIO  │
                │ :9000   │
                └─────────┘


 

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


This setup provides a more robust and scalable solution for managing machine learning models in a
production-oriented MLOps architecture.



# ------------------------------- Environment Variables -----------------------------


.env                         Settings
────────────────────────────────────────────
APP_NAME              →      app_name
APP_VERSION           →      app_version
ENVIRONMENT           →      environment
DEBUG                 →      debug

MLFLOW_TRACKING_URI   →      mlflow_tracking_uri

MODEL_NAME            →      model_name
MODEL_ALIAS           →      model_alias

AWS_ACCESS_KEY_ID     →      aws_access_key_id
AWS_SECRET_ACCESS_KEY →      aws_secret_access_key

MLFLOW_S3_ENDPOINT_URL →     mlflow_s3_endpoint_url

API_PREFIX            →      api_prefix
LOG_LEVEL             →      log_level 


                 ┌────────────────────┐
                 │   .env (local)     │
                 │                    │
                 │ credentials        │
                 └─────────┬──────────┘
                           ↓
                    Pydantic Settings
                           ↓
                ┌──────────┴──────────┐
                ↓                     ↓
             FastAPI              MLflow
                ↓                     ↓
             config              model URI 


# ------------------------------- Dockernize complete application -----------------------------

                    ┌──────────────┐
                    │   FastAPI    │
                    │   :8001      │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   MLflow     │
                    │   :5000      │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       ┌──────────────┐          ┌──────────────┐
       │ PostgreSQL   │          │    MinIO     │ 
       │   metadata   │          │  artifacts   │
       └──────────────┘          └──────────────┘     





# ------------------------------- Monitoring grafana + prometheus -----------------------------

PrompQL to see the metrics for the FastAPI service:

{job="fastapi"} 
                                              



                         ┌─────────────────────┐
                         │      Client         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      FastAPI        │
                         │                     │
                         │ /predictions        │
                         │ /health             │
                         │ /health/ready       │
                         │ /model              │
                         │ /metrics            │
                         └───────┬─────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
             ┌──────────────┐         ┌──────────────┐
             │ MLflow    │            │  Prometheus  │
             │ Model        │         │    :9090     │
             │ Registry     │         └──────┬───────┘
             └──────────────┘                │
                                             ▼
                                      ┌──────────────┐
                                      │   Grafana    │
                                      │    :3000     │
                                      └──────────────┘

       # Test dev mode with uvicorn and curl

        uvicorn app.api.main:app --reload --port 8001
       curl http://localhost:8001/api/v1/metrics
       curl http://localhost:8001/api/v1/health

       curl http://localhost:8001/api/v1/model 


       prometheus ui to see metrics 

       http://localhost:9090/targets 

       docker exec prometheus wget -qO- http://fastapi:8000/metrics 


       # Grafana dashboard to see metrics


# Recommended dashboard

At this point, I'd structure your Grafana dashboard like this:


┌────────────────────┬────────────────────┬────────────────────┐
│ FastAPI Status     │ Total Predictions  │ Prediction Errors  │
│       UP            │        12          │         0         │
└────────────────────┴────────────────────┴────────────────────┘

┌────────────────────┬────────────────────┬────────────────────┐
│ Successful         │ Success Rate       │ Predictions / Min  │
│ Predictions: 12    │       100%         │                    │
└────────────────────┴────────────────────┴────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│                Prediction Requests / Minute                   │
│                         📈                                    │
└───────────────────────────────────────────────────────────────┘

┌─────────────────────────────┬─────────────────────────────────┐
│ Average Prediction Latency  │ P95 Prediction Latency         │
│            ~54 ms           │                                 │
└─────────────────────────────┴─────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│                    Prediction Latency                         │
│                         📈                                    │
└───────────────────────────────────────────────────────────────┘

┌────────────────────────────────┬──────────────────────────────┐
│ Predictions by Class            │ FastAPI Memory               │
│                                │                              │
│ malignant ████████████ 12      │          📈                  │
│ benign                         │                              │
└────────────────────────────────┴──────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│                       CPU Usage                               │
│                         📈                                    │
└───────────────────────────────────────────────────────────────┘