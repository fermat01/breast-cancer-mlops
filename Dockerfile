FROM python:3.9-slim

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

# Copy the trained model file into the container
COPY rf_model.joblib /app/

EXPOSE 80 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
