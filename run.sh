#!/bin/bash
set -e

# Create virtualenv if not exists
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

# Activate virtualenv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r app/requirements.txt

# Run training to generate rf_model.joblib
python app/model.py


# Deactivate virtualenv
deactivate

# Build docker image including the trained model
docker build -t breast-cancer-mlapp .

# Remove old container if exists
docker rm -f breast-cancer-mlapp || true

# Run new container exposing ports
docker run -d --name breast-cancer-mlapp -p 8000:80 -p 8001:8001 breast-cancer-mlapp

echo "Container is running. Access API at http://localhost:8000/health"
