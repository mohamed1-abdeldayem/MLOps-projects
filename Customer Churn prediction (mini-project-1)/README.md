<div align="center">

<br/>

# Customer Churn Prediction — MLOps Service

**An end-to-end machine learning service that predicts customer churn, from raw data to a containerized, ONNX-served API.**

Reproducible training pipeline · ONNX-optimized inference · FastAPI backend · Static frontend · Docker Compose orchestration

<br/>

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.14x-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![uv](https://img.shields.io/badge/Package%20Manager-uv-DE5FE9)
![License](https://img.shields.io/badge/License-MIT-green)

<br/>

[Overview](#overview) · [Architecture](#architecture) · [Pipeline Stages](#pipeline-stages) · [Quick Start](#quick-start) · [API Reference](#api-reference) · [Testing](#testing) · [Configuration](#configuration) · [Tech Stack](#tech-stack)

<br/>

</div>

---

## Overview

This project is a **complete, containerized MLOps mini-pipeline** for predicting whether a telecom customer is likely to churn, built on the classic [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset.

It takes the model lifecycle beyond a notebook: raw data is cleaned and encoded, a Logistic Regression classifier is tuned via cross-validated grid search, the winning pipeline is exported to **ONNX** for fast, dependency-light inference, and the whole system — API and frontend — is packaged into two Docker images orchestrated with Docker Compose.

| Capability | Details |
|---|---|
| **Reproducible training** | A single `ModelTrainer` class handles loading, preprocessing, cross-validated tuning, evaluation, and saving |
| **Optimized serving** | The trained scikit-learn pipeline is converted to **ONNX** and served with `onnxruntime` instead of `pickle` |
| **Typed API** | FastAPI + Pydantic validate every incoming customer record before it reaches the model |
| **Automated tests** | Pytest suite covering the training pipeline, the inference layer, and the HTTP API |
| **Full containerization** | Separate Docker images for the FastAPI backend and the static frontend, wired together with `docker-compose.yml` |
| **Environment-driven config** | Data and model paths are injected via `.env` / `pydantic-settings`, not hardcoded |

> **Who is this for?** Anyone learning how to take a scikit-learn model from a notebook to a served, containerized API — including preprocessing consistency, ONNX export, and clean API design.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                     Customer Churn Prediction — Architecture              │
└───────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌───────────────────────┐     ┌────────────────────┐
  │  DATA LAYER  │     │   TRAINING (offline)   │     │   MODEL ARTIFACTS  │
  │              │     │                        │     │                    │
  │ Telco Churn  │     │ training/training.py   │     │ models/            │
  │ CSV ─────────┼────►│  ├─ load + clean        │     │ ├─ model.pkl       │
  │              │     │  ├─ one-hot encode      │────►│ ├─ model.onnx      │
  │ data/raw     │     │  ├─ GridSearchCV        │     │ └─ feature_names   │
  │ data/processed│    │  │   (LogReg + CV)      │     │     .json          │
  └──────────────┘     │  ├─ evaluate (Acc/P/R/F1)│    └─────────┬──────────┘
                        │  └─ save_model()        │              │
                        │                        │              │
                        │ training/export_onnx.py │              │
                        │  pickle ──► ONNX ────────┼──────────────┘
                        └───────────────────────┘

                    ┌─────────────────────────────────────────┐
                    │     BACKEND  — FastAPI (:8000)           │
                    │     src/mlops_churn_api/                 │
                    │                                          │
                    │  lifespan → load ONNX model once          │
                    │  POST /predict   churn label + probability│
                    │  GET  /health    model_loaded status       │
                    │  GET  /          liveness message          │
                    └──────────────────┬───────────────────────┘
                                       │ REST (CORS-enabled)
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │     FRONTEND — Static site (nginx :80)   │
                    │     frontend/index.html + script.js       │
                    │                                          │
                    │  Customer form → POST /predict → result   │
                    └─────────────────────────────────────────┘

              docker-compose.yml orchestrates backend + frontend
```

---

## Pipeline Stages

### 1 — Data Preprocessing

`training/training.py → ModelTrainer.pre_processing()` cleans the raw Telco Churn CSV before anything else runs:

| Step | What happens |
|---|---|
| Target encoding | `Churn` mapped from `Yes` / `No` to `1` / `0` |
| Type coercion | `TotalCharges` coerced to numeric (raw source stores it as text) |
| Row filtering | Rows with `tenure == 0` (brand-new, unlabeled accounts) are dropped |
| Column pruning | `customerID` and `PhoneService` are dropped as non-predictive |
| Encoding | Remaining categoricals are one-hot encoded (`pd.get_dummies`, `drop_first=True`) |
| Artifact | The exact column order is persisted to `models/feature_names.json` so inference can reindex incoming requests to match training features |

The cleaned dataset is also written to `data/processed/Telco-Customer-Churn-processed.csv`.

### 2 — Model Training & Selection

`ModelTrainer.create_model()` builds a `scikit-learn` **Pipeline** of `StandardScaler → LogisticRegression`, tuned with `GridSearchCV` over three solver/penalty families and a `StratifiedKFold(5)` cross-validation, optimizing for **ROC-AUC**:

| Solver | Penalty | `C` grid |
|---|---|---|
| `liblinear` | `l1`, `l2` | `np.logspace(-4, 4, 9)` |
| `lbfgs` | `l2` | `np.logspace(-4, 4, 9)` |
| `saga` | `elasticnet` (`l1_ratio` ∈ {0.25, 0.5, 0.75}) | `np.logspace(-4, 4, 9)` |

The best estimator (by mean CV ROC-AUC) is evaluated on a held-out 20% stratified test split, logging **Accuracy, Precision, Recall, and F1** for the churn class, then serialized with `joblib`.

### 3 — ONNX Export

`training/export_onnx.py` loads the trained `pickle` pipeline and converts it to **ONNX** via `skl2onnx`, so the API never needs `scikit-learn` at inference time — only the lightweight `onnxruntime`.

### 4 — Model Serving

`src/mlops_churn_api/inference.py` loads the ONNX model and `feature_names.json` once at API startup. Each request is:

1. Validated against the `ChurnInput` Pydantic schema
2. One-hot encoded and **reindexed** to the exact training-time feature order (missing dummy columns filled with `0`)
3. Run through the ONNX session to return the predicted class and its probability

### 5 — API & Frontend

A FastAPI app (`src/mlops_churn_api/main.py`) exposes the model behind three endpoints (see [API Reference](#api-reference)), with CORS opened for a plain HTML/CSS/JS frontend (`frontend/`) served separately by nginx.

### 6 — Testing

Three pytest modules cover the pipeline end to end — see [Testing](#testing).

---

## Project Structure

```
Customer-Churn-Prediction-mini-project-1-/
├── data/
│   ├── raw/                          # Source Telco Churn CSV
│   └── processed/                    # Cleaned CSV written during training
├── src/
│   └── mlops_churn_api/
│       ├── main.py                   # FastAPI app: /, /health, /predict
│       ├── inference.py              # ONNX model loading + prediction
│       ├── schemas.py                # ChurnInput Pydantic model
│       ├── config.py                 # pydantic-settings, .env driven
│       └── __init__.py
├── training/
│   ├── training.py                   # ModelTrainer: load → preprocess → tune → evaluate → save
│   ├── export_onnx.py                # ExportToOnnx: pickle → ONNX
│   └── __init__.py
├── models/
│   ├── model.pkl                     # Trained scikit-learn pipeline (generated)
│   ├── model.onnx                    # ONNX-exported model (generated)
│   └── feature_names.json            # Training-time feature order (generated)
├── notebooks/
│   └── 01_eda.ipynb                  # Exploratory data analysis
├── frontend/
│   ├── index.html                    # Customer data input form
│   ├── script.js                     # Calls the /predict endpoint
│   ├── style.css
│   └── Dockerfile                    # nginx:alpine static image
├── tests/
│   ├── test_training.py              # ModelTrainer unit tests
│   ├── test_inference.py             # Inference/preprocessing tests
│   └── test_api.py                   # FastAPI endpoint tests
├── logs/                             # Training/evaluation logs
├── docker-compose.yml                # Orchestrates backend + frontend
├── Dockerfile                        # Backend image (python:3.12-slim + uv)
├── pyproject.toml                    # uv-managed dependencies
├── uv.lock
├── .env.example                      # DATA_PATH / MODEL_PICKLE_PATH / MODEL_ONNX_PATH
└── LICENSE                           # MIT
```

---

## Quick Start

### Prerequisites

```
Python 3.12+   git   uv   docker   docker-compose
```

### 1. Clone & install

```bash
git clone -b mlops/mini-project-1 https://github.com/mohamed1-abdeldayem/MLOps-projects.git
cd Customer-Churn-Prediction-mini-project-1-
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Then fill in the three required paths (relative to the project root):

```
DATA_PATH=data/raw/Telco-Customer-Churn.csv
MODEL_PICKLE_PATH=models/model.pkl
MODEL_ONNX_PATH=models/model.onnx
```

> Download the Telco Customer Churn CSV and place it under `data/raw/` before training.

### 3. Train the model

```bash
uv run python training/training.py
```

This preprocesses the data, runs the grid search, logs evaluation metrics (Accuracy / Precision / Recall / F1), and saves `models/model.pkl`.

### 4. Export to ONNX

```bash
uv run python training/export_onnx.py
```

### 5. Run the API locally

```bash
uv run uvicorn mlops_churn_api.main:app --app-dir src --host 0.0.0.0 --port 8000 --reload
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "gender": "Male", "SeniorCitizen": 0, "partner": "Yes", "Dependents": "No",
        "tenure": 12, "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No",
        "Contract": "One year", "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check", "MonthlyCharges": 70.5, "TotalCharges": 846.0
      }'
```

### 6. Run the test suite

```bash
uv run pytest tests/ -v
```

### 7. Full Docker stack

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Backend API | http://localhost:8000 |
| Frontend | http://localhost:5500 |

---

## API Reference

Interactive docs auto-generated by FastAPI at [http://localhost:8000/docs](http://localhost:8000/docs).

---

### `POST /predict` — Predict churn

**Request body** (`ChurnInput`)

| Field | Type | Description |
|---|---|---|
| `gender` | string | `Male` / `Female` |
| `SeniorCitizen` | int | `0` or `1` |
| `partner` | string | `Yes` / `No` |
| `Dependents` | string | `Yes` / `No` |
| `tenure` | int | Months as a customer |
| `MultipleLines` | string | `Yes` / `No` / `No phone service` |
| `InternetService` | string | `DSL` / `Fiber optic` / `No` |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | string | `Yes` / `No` / `No internet service` |
| `Contract` | string | `Month-to-month` / `One year` / `Two year` |
| `PaperlessBilling` | string | `Yes` / `No` |
| `PaymentMethod` | string | e.g. `Mailed check`, `Electronic check` |
| `MonthlyCharges` | float | Current monthly charge |
| `TotalCharges` | float | Total charges to date |

**Response**

```json
{
  "prediction": 0,
  "probability": 0.8123
}
```

`prediction` is `1` if the customer is predicted to churn, else `0`. `probability` is the model's confidence in the predicted class.

---

### `GET /health` — Health check

```bash
curl http://localhost:8000/health
# {"status": "healthy", "model_loaded": true}
```

---

### `GET /` — Liveness message

```bash
curl http://localhost:8000/
# {"message": "Churn API is running"}
```

---

## Testing

```
tests/
├── test_training.py    — ModelTrainer: data loading, preprocessing, splitting,
│                          grid-search config, training, evaluation logging, saving
├── test_inference.py   — Inference: feature reindexing, end-to-end prediction sanity
└── test_api.py         — FastAPI TestClient: /, /health, /predict
```

```bash
uv run pytest tests/ -v
```

---

## Configuration

All paths are supplied through environment variables loaded via `pydantic-settings` (`src/mlops_churn_api/config.py`), resolved relative to the project root:

```bash
# .env
DATA_PATH=data/raw/Telco-Customer-Churn.csv
MODEL_PICKLE_PATH=models/model.pkl
MODEL_ONNX_PATH=models/model.onnx
```

The training pipeline reads `DATA_PATH` and writes `MODEL_PICKLE_PATH`; the API reads `MODEL_ONNX_PATH` at startup to load the ONNX inference session.

---

## Docker

The backend uses a slim, `uv`-driven build; the frontend is a plain static nginx image.

**Backend (`Dockerfile`)**
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN pip install uv
COPY src ./src
COPY models ./models
RUN uv sync --frozen
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "mlops_churn_api.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
```

**Frontend (`frontend/Dockerfile`)**
```dockerfile
FROM nginx:alpine
COPY . /usr/share/nginx/html
EXPOSE 80
```

**Orchestration (`docker-compose.yml`)** builds and wires both services, with the frontend depending on the backend and reaching it over the network on port `8000`.

```bash
docker-compose up --build
```

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Dataset | Telco Customer Churn (Kaggle) | Binary churn classification |
| ML | scikit-learn — `StandardScaler` + `LogisticRegression` | Feature scaling + classification |
| Model selection | `GridSearchCV` + `StratifiedKFold` (ROC-AUC) | Hyperparameter tuning across 3 solver families |
| Model export | `skl2onnx` | Converts the fitted pipeline to ONNX |
| Serving runtime | `onnxruntime` | Fast, scikit-learn-free inference |
| API | FastAPI + Pydantic + `pydantic-settings` | Typed REST endpoint, env-driven config |
| Frontend | Static HTML / CSS / JS | Customer data form, calls the API directly |
| Testing | Pytest + FastAPI `TestClient` | Training, inference, and API test coverage |
| Package management | `uv` | Dependency resolution and locking |
| Containerization | Docker (backend + frontend) + Docker Compose | Reproducible local orchestration |

---

## Contributing

Pull requests are welcome. For larger changes, please open an issue first to discuss what you'd like to change.

```bash
git checkout -b feature/your-feature
# make changes
uv run pytest tests/ -v
git commit -m "feat: add your feature"
git push origin feature/your-feature
# open a pull request
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built by [mohamed1-abdeldayem](https://github.com/mohamed1-abdeldayem) · [Repository](https://github.com/mohamed1-abdeldayem/Customer-Churn-Prediction-mini-project-1-)

</div>