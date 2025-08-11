# breast-cancer-mlops


![GitHub](https://img.shields.io/github/license/fermat01/breast-cancer-mlops?style=flat)
![GitHub top language](https://img.shields.io/github/languages/top/fermat01/breast-cancer-mlops?style=flat)
![GitHub language count](https://img.shields.io/github/languages/count/fermat01/breast-cancer-mlops?style=flat)
![GitHub last commit](https://img.shields.io/github/last-commit/fermat01/breast-cancer-mlops?style=flat)
![ViewCount](https://views.whatilearened.today/views/github/fermat01/breast-cancer-mlops.svg?cache=remove)



An end-to-end Machine Learning Operations (MLOps) project for Breast Cancer classification using the Wisconsin dataset. This repository includes model training with MLflow tracking, FastAPI-based REST API serving for predictions, Prometheus metrics integration, Docker containerization, and an automated CI/CD pipeline deploying on AWS ECS (Fargate).

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Technology Stack](#technology-stack)  
- [Installation](#installation)  
- [Usage](#usage)  
- [API Endpoints](#api-endpoints)  
- [Monitoring](#monitoring)  
- [CI/CD Pipeline](#cicd-pipeline)  
- [Deployment](#deployment)  
- [Development](#development)  
- [Contributing](#contributing)  
- [License](#license)

---

## Project Overview

This MLOps project provides a complete workflow for building, serving, monitoring, and deploying a machine learning model that classifies breast cancer using the Wisconsin dataset. The model is trained using scikit-learn's Random Forest implementation, tracked with MLflow, and served via FastAPI. Prometheus monitoring is integrated for observability, and the application is containerized with Docker and deployable via an automated GitHub Actions-based CI/CD pipeline targeting AWS ECS with Fargate.

---

## Features

- **Training:** Model training with automated logging of parameters and metrics using MLflow.  
- **Serving:** FastAPI RESTful API serving predictions with input validation via Pydantic, containerized for cloud deployment.  
- **Monitoring:** Prometheus metrics exposed for request counts and latency histogram, integrated with AWS CloudWatch for observability in ECS.  
- **Containerization:** Dockerfile for creating lightweight containers, managed via ECS Fargate for serverless container orchestration and automatic scaling.  
- **CI/CD:** Automated pipeline with GitHub Actions for building, testing, pushing Docker images to Amazon ECR, and deploying to AWS ECS Fargate services with zero downtime deployments.  
- **Security:** Sensitive credentials stored securely through GitHub Secrets and integrated with AWS IAM roles and AWS Systems Manager Parameter Store or Secrets Manager for runtime secrets management.

---

## Technology Stack

| Component           | Technology / Library                                               |
|---------------------|--------------------------------------------------------------------|
| Model Training      | scikit-learn, MLflow                                               |
| API Framework       | FastAPI, Pydantic                                                  |
| Metrics & Monitoring| Prometheus client, AWS CloudWatch integration                     |
| Containerization    | Docker, Amazon ECR (Elastic Container Registry), ECS Fargate       |
| CI/CD               | GitHub Actions, GitHub Secrets, AWS CLI for ECS deployment        |
| Cloud Deployment    | AWS ECS (Fargate) — serverless container orchestration with scaling|
| Language & Tools    | Python 3.8+, numpy, joblib                                         
        

---

## Installation

### Prerequisites

- Python 3.8+  
- Docker & Docker Compose (optional)  
- Git

### Local Setup

1. Clone the repository:

```git clone https://github.com/fermat01/breast-cancer-mlops.git```


``` cd breast-cancer-mlops ```

---

## Usage

- Send POST requests to `/predict` endpoint with JSON body containing 30 breast cancer feature values.  
- Access `/health` endpoint to check API health status.

- Example prediction request body:

```
{
"mean_radius": 14.5,
"mean_texture": 20.3,
"mean_perimeter": 90.2,
"mean_area": 600.1,
"mean_smoothness": 0.1,
...
"worst_fractal_dimension": 0.3
}
```


(Include all required features exactly as specified in the `BreastCancerFeatures` Pydantic model.)

---

## API Endpoints

| Method | Endpoint     | Description                        |
|--------|--------------|----------------------------------|
| POST   | `/predict`   | Get prediction for breast cancer |
| GET    | `/health`    | Check API health status           |

---

## Monitoring

- Prometheus metrics exposed on port **8001**.  
- Metrics include request count and request latency histogram.  
- Configure Prometheus server to scrape the application metrics for observability or integrate with AWS CloudWatch for centralized monitoring.

---

## CI/CD Pipeline

-  GitHub Actions used to:  
  - Build and test Docker image.  
  - Push image to Amazon Elastic Container Registry (ECR) (credentials managed via GitHub Secrets).  
  - Deploy to AWS ECS Fargate service with zero downtime updates using AWS CLI or GitHub Actions ECS deploy action.  
- Environment variables and secrets are managed safely:  
  - Non-sensitive configuration via `.env`.  
  - Sensitive tokens and keys via GitHub Secrets and AWS Systems Manager Parameter Store or Secrets Manager.


---

## Deployment

### Deploy on AWS ECS using Fargate 

**Prerequisites:**  
- AWS CLI configured with appropriate IAM permissions.  
- Docker image pushed to Amazon ECR.

**Steps:**  
1. Create an ECS cluster with Fargate launch type in the AWS Console or via CLI.  
2. Define a Task Definition specifying your container image, CPU, memory requirements, and port mappings (8000 for API, 8001 for metrics).  
3. Create a Service linked to the ECS cluster and Task Definition, configuring an Application Load Balancer to route traffic.  
4. Setup Auto Scaling policies based on CPU, memory, or request load.  
5. Use GitHub Actions to build, push the Docker image, and update ECS Services for seamless deployment.

---

## Development

- Helper functions in `utils.py`.  
- Customize model logic in `model.py`.  
- Extend API in `main.py`.  
- MLflow UI for experiment tracking (`mlflow ui`).  
- Docker Compose available for local multi-container setups.

---

## Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/new-feature`)  
3. Commit your changes (`git commit -m 'Add feature'`)  
4. Push to your branch (`git push origin feature/new-feature`)  
5. Open a Pull Request

Please ensure your code adheres to project style and passes tests.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or support, please open an issue or contact me

---

*Happy MLOps development and breast cancer prediction!*

