# breast-cancer-mlops


![GitHub](https://img.shields.io/github/license/fermat01/breast-cancer-mlops?style=flat)
![GitHub top language](https://img.shields.io/github/languages/top/fermat01/breast-cancer-mlops?style=flat)
![GitHub language count](https://img.shields.io/github/languages/count/fermat01/breast-cancer-mlops?style=flat)
![GitHub last commit](https://img.shields.io/github/last-commit/fermat01/breast-cancer-mlops?style=flat)
![ViewCount](https://views.whatilearened.today/views/github/fermat01/breast-cancer-mlops.svg?cache=remove)

An end-to-end Machine Learning Operations (MLOps) project for Breast Cancer classification using the Wisconsin dataset. This repository includes model training with MLflow tracking, FastAPI-based REST API serving for predictions, Prometheus metrics integration, Docker containerization, and an automated CI/CD pipeline deploying on AWS EC2.

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

This MLOps project provides a complete workflow for building, serving, monitoring, and deploying a machine learning model that classifies breast cancer using the Wisconsin dataset. The model is trained using scikit-learn's Random Forest implementation, tracked with MLflow, and served via FastAPI. Prometheus monitoring is integrated for observability, and the application is containerized with Docker and deployable via an automated GitHub Actions-based CI/CD pipeline.

---

## Features

- **Training:** Model training with automated logging of parameters and metrics using MLflow.  
- **Serving:** FastAPI RESTful API serving predictions with input validation via Pydantic.  
- **Monitoring:** Prometheus metrics exposed for request counts and latency histogram.  
- **Containerization:** Dockerfile for creating lightweight containers, optionally managed with Docker Compose.  
- **CI/CD:** Automated pipeline with GitHub Actions for building, testing, pushing Docker images, and deploying to AWS EC2 instances.  
- **Security:** Sensitive credentials stored securely through GitHub Secrets.

---

## Technology Stack

| Component           | Technology / Library                       |
|---------------------|------------------------------------------|
| Model Training      | scikit-learn, MLflow                      |
| API Framework       | FastAPI, Pydantic                         |
| Metrics & Monitoring| Prometheus client                         |
| Containerization    | Docker, Docker Compose                     |
| CI/CD               | GitHub Actions, GitHub Secrets            |
| Cloud Deployment    | AWS EC2                                   |
| Language & Tools    | Python 3.8+, numpy, joblib                |

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
- Configure Prometheus server to scrape your application metrics for observability.

---

## CI/CD Pipeline

- Uses GitHub Actions to:  
  - Build and test Docker image.  
  - Push image to Docker Hub (credentials in GitHub Secrets).  
  - SSH deploy container to AWS EC2 instance securely.  
- Environment variables and secrets managed safely:  
  - Non-sensitive config via `.env:`  
  - Sensitive tokens & keys via GitHub Secrets.

---

## Deployment

- Dockerized application can be deployed on:  
  - AWS EC2 (via CI/CD pipeline or manual Docker commands).  
  - Local or cloud servers supporting Docker.  
- Ports exposed:  
  - 8000: API server  
  - 8001: Prometheus metrics

---

## Development

- Helper functions in `utils.py`.  
- Customize model logic in `model.py`.  
- Extend API in `main.py`.  
- MLflow UI for experiment tracking (`mlflow ui`).  
-  Docker Compose for managing multi-container setups if needed.

---

## Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/new-feature`)  
3. Commit your changes (`git commit -m 'Add feature'`)  
4. Push to your branch (`git push origin feature/new-feature`)  
5. Open a Pull Request

Please ensure code adheres to project style and passes tests.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or support, please open an issue or contact me

---

*Happy MLOps development and breast cancer prediction!*

