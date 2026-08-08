


./scripts/train_and_promote.sh


### What the script does

The important part is the sequence:



Docker services
      │
      ▼
PostgreSQL + MLflow + MinIO
      │
      ▼
Export environment variables
      │
      ▼
python -m training.train
      │
      ├── Load dataset
      ├── Validate
      ├── Split
      ├── Preprocess
      ├── Train RandomForest
      ├── Evaluate
      ├── Log metrics → PostgreSQL
      └── Log model → MinIO
      │
      ▼
python -m training.assign_model_alias
      │
      ▼
breast-cancer-classifier
      │
      └── champion → latest approved model version
      │
      ▼
FastAPI
      │
      └── models:/breast-cancer-classifier@champion 

The key distinction is that PostgreSQL does not store the model itself.

We have :

PostgreSQL
    │
    ├── experiments
    ├── runs
    ├── metrics
    ├── parameters
    ├── registered models
    └── model versions 

and 

MinIO
    │
    └── mlflow-artifacts/
            │
            └── model artifacts
                    ├── MLmodel
                    ├── model.pkl
                    ├── conda.yaml
                    ├── python_env.yaml
                    └── requirements.txt