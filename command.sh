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