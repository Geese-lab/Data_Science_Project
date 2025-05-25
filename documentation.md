# NYC Taxi Fare Prediction Project Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Setup & Installation](#setup--installation)
4. [Data Pipeline](#data-pipeline)
5. [Feature Engineering](#feature-engineering)
6. [Modeling](#modeling)

   * [Supervised Models](#supervised-models)
   * [Ensemble Methods](#ensemble-methods)
   * [Deep Learning](#deep-learning)
   * [Clustering](#clustering)
7. [Evaluation & Validation](#evaluation--validation)
8. [Deployment](#deployment)
9. [Monitoring & Maintenance](#monitoring--maintenance)
10. [Contributing Guidelines](#contributing-guidelines)
11. [References](#references)

---

## 1. Project Overview

The NYC Taxi Fare Prediction project aims to predict taxi trip fares using the 2019 NYC Taxi Trips dataset (103M records). It addresses two tasks:

* **Regression**: Predict continuous fare amounts.
* **Classification**: Categorize fares into four discrete classes.

Workflow follows a six-phase data science lifecycle:

1. Data Cleansing
2. Exploratory Data Analysis
3. Feature Engineering
4. Model Development
5. Operationalization
6. Ethical Governance

---

## 2. Repository Structure

```bash
├── data/                  # Raw and processed data files
│   ├── raw/               # Original dataset CSVs
│   └── processed/         # Cleaned and feature-engineered data
├── notebooks/             # Jupyter notebooks for EDA and prototyping
├── src/                   # Source code modules
│   ├── preprocessing/     # Data cleaning and preprocessing scripts
│   ├── features/          # Feature engineering functions
│   ├── models/            # Model training and evaluation scripts
│   ├── deployment/        # FastAPI endpoints and Dockerfiles
│   └── utils/             # Utility functions
├── tests/                 # Unit and integration tests
├── docs/                  # Additional documentation (this file)
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container specification
├── docker-compose.yml     # Local environment orchestration
├── README.md              # Project summary and quick start
└── LICENSE                # Project license
```

---

## 3. Setup & Installation

### Prerequisites

* Python 3.8+
* Docker & Docker Compose
* AWS CLI (for cloud deployment)

### Local Environment

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-org/nyc-taxi-fare-prediction.git
   cd nyc-taxi-fare-prediction
   ```
2. **Create and activate virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
4. **Run tests**:

   ```bash
   pytest
   ```

---

## 4. Data Pipeline

### Raw Data Ingestion

* **Location**: `data/raw/`
* **Format**: CSV files for training and testing sets

### Preprocessing Steps (`src/preprocessing`)

1. **Duplicate Removal**: Drop exact duplicates.
2. **Missing Value Handling**: Imputation or row dropping.
3. **Outlier Detection**: IQR-based filtering.
4. **Datetime Parsing**: Convert timestamp strings to `datetime`.

**Usage**:

```python
from src.preprocessing.clean import clean_data
train_df = clean_data("data/raw/train.csv")
```

---

## 5. Feature Engineering

**Location**: `src/features`

Key engineered features:

* Trip duration (minutes)
* Pickup hour & day of week
* Weekend flag
* Distance per passenger
* Fare per mile
* Time of day category
* Tip flag
* Airport trip indicator

**Usage**:

```python
from src.features.build_features import add_features
df = add_features(train_df)
```

---

## 6. Modeling

### Supervised Models (`src/models/supervised.py`)

* **Regression**: Linear, Ridge, Lasso, Random Forest, Gradient Boosting
* **Classification**: Random Forest, Gradient Boosting

Models are configurable via YAML or CLI arguments, with built-in cross-validation.

### Ensemble Methods (`src/models/ensemble.py`)

* **Bagging**: `VotingRegressor` / `VotingClassifier`
* **Boosting**: XGBoost / LightGBM wrappers

### Deep Learning (`src/models/deep_learning.py`)

* **Architecture**: Fully connected MLP for both regression and classification
* **Training**: 50 epochs with early stopping
* **Imbalance Handling**: Class weighting in loss function

### Clustering (`src/models/clustering.py`)

* **K-Means**: Silhouette analysis over multiple K values
* **DBSCAN**: Grid search over `eps` and `min_samples`

---

## 7. Evaluation & Validation

* **Metrics**:

  * Regression: MAE, RMSE
  * Classification: Accuracy, F1-score
* **Cross-validation**: 5-fold
* **Hypothesis Testing**: Two-sample t-test for group comparisons

Results and plots are stored under `reports/` and `figures/`.

---

## 8. Deployment

### Containerization

* **Dockerfile**: Builds a FastAPI server image
* **docker-compose.yml**: Orchestrates API plus optional database

### AWS SageMaker

* **Endpoints**: Models deployed as Docker containers
* **CI/CD**: GitHub Actions triggers on `main` branch
* **Monitoring**: Prometheus & Grafana dashboards

### API Example

**Request**:

```http
POST /predict_fare
Content-Type: application/json

{
  "pickup_datetime": "2025-03-01T08:30:00",
  "PULocationID": 130,
  // ... other fields
}
```

**Response**:

```json
{
  "fare": 12.34,
  "confidence": 0.87
}
```

---

## 9. Monitoring & Maintenance

* **Retraining**: Monthly full retrains, weekly incremental updates
* **Alerts**:

  * MAE > \$8 triggers Slack notification
  * Data drift detection via Evidently AI
* **Logging**: Structured logging with `structlog`

---

## 10. Contributing Guidelines

1. **Issue Tracking**: Use GitHub Issues
2. **Branching**: `feature/*`, `bugfix/*`, `hotfix/*`
3. **Pull Requests**:

   * Describe changes and tests
   * Assign a reviewer
   * Ensure CI passing
4. **Code Style**: Follow PEP8; auto-format with `black`
5. **Testing**: Maintain ≥80% coverage

---

## 11. References

1. NYC Taxi & Limousine Commission. *2019 NYC Taxi Trips Dataset*.
2. Kuhn, M., & Johnson, K. (2019). *Feature Engineering and Selection*. Chapman and Hall/CRC.
3. Chollet, F. (2017). *Deep Learning with Python*. Manning Publications.
