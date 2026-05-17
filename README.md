# 🏥 MLOps Pipeline — Breast Cancer Prediction

[![CI Pipeline](https://github.com/pranavi11111/mlops-project/actions/workflows/ci.yml/badge.svg)](https://github.com/pranavi11111/mlops-project/actions)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Minikube-326CE5)
![MLflow](https://img.shields.io/badge/MLflow-2.11-orange)

> An end-to-end production-grade MLOps pipeline built for the AiSPRY evaluation assignment. Implements all 12 MLOps components using industry-standard tools and best practices.

## 🌐 Live Demo

| Component | URL |
|-----------|-----|
| 🖥️ React Frontend | https://mlops-vision.lovable.app |
| 🔗 GitHub Repo | https://github.com/pranavi11111/mlops-project |

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 96.49% |
| Precision | 95.89% |
| Recall | 98.59% |
| F1 Score | 97.22% |

Dataset: Breast Cancer Wisconsin from sklearn — 569 samples, 30 features
Model: Random Forest Classifier with 200 estimators
Task: Binary classification — Benign vs Malignant

## 🛠️ Complete MLOps Stack

### 1. Data Lineage — OpenLineage + Marquez
- Tracks complete data pipeline lineage
- 5 lineage events: load, validate, train, evaluate, serve
- Visual lineage graph in lineage_reports/lineage_graph.html

### 2. Versioning — DVC
- Data and model versioning
- Tracks dataset.csv and model.pkl

### 3. Data Quality — Great Expectations
- Automated data validation on every run
- Validates feature ranges, nulls, data types

### 4. Feature Store — Feast
- 6 patient features managed consistently
- mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_concavity

### 5. Experiment Tracking — MLflow
- Tracks all training runs with metrics
- Model registry with versioning

### 6. Orchestration — Docker + Kubernetes
- Containerized FastAPI on Minikube
- Deployment and service manifests in k8s/

### 7. Deployment — FastAPI + ReactJS
- FastAPI at /predict endpoint
- React frontend: https://mlops-vision.lovable.app
- Real-time predictions with confidence scores

### 8. Monitoring — Evidently AI
- Data drift detection
- HTML reports in monitoring/reports/

### 9. Infrastructure Monitoring — Grafana + Prometheus
- Real-time API metrics with Prometheus
- Grafana dashboards for performance monitoring

### 10. CI — GitHub Actions
- Automated pipeline on every push
- Tests, linting, Docker build validation

### 11. CD — ArgoCD
- GitOps continuous deployment
- Auto-syncs from GitHub to Kubernetes
- Status: Healthy and Synced

### 12. Prompt Testing — Promptfoo
- 2 automated API tests
- 100% pass rate

## 📁 Project Structure
```
mlops-project/
├── app/                 FastAPI application
├── model/               Training pipeline
├── feature_store/       Feast feature store
├── k8s/                 Kubernetes manifests
├── monitoring/          Drift monitoring
├── lineage_reports/     OpenLineage reports
├── gx/                  Great Expectations
├── .dvc/                DVC versioning
├── .github/workflows/   GitHub Actions CI
├── argocd-app.yaml      ArgoCD config
├── promptfooconfig.yaml Promptfoo tests
├── docker-compose.yml   Docker services
├── Dockerfile           Container definition
└── prometheus.yml       Prometheus config
```
## 🚀 How to Run

### Train model
cd model
python train.py

### Start API
cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

### Start all services
docker-compose up -d

### Deploy to Kubernetes
minikube start --driver=docker
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

### Run tests
npx promptfoo eval

## 👩‍💻 Author

Doragolla Pranavi
GitHub: https://github.com/pranavi11111

Built for AiSPRY MLOps Evaluation Assignment