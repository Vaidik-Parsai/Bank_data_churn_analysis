# 🏦 Bank Customer Churn: Analytics & Production ML Serving Service

An interview-ready, end-to-end Machine Learning and Business Intelligence solution for retail banking customer retention. This repository unites exploratory analytics, ML experimentation notes, and a production-grade Scikit-Learn pipeline deployed via FastAPI and Docker.

---

## 📌 Problem Statement

Customer attrition is one of the costliest operational challenges in retail banking. Acquiring new customers costs 5× to 7× more than retaining existing accounts. In an imbalanced customer portfolio where roughly **20.37% of customers churn** (1 in every 5 customers), standard off-the-shelf models fail by optimizing for raw accuracy while missing the vast majority of leaving customers.

This project delivers:
1. **Business Intelligence & Analytics**: Interactive Power BI dashboard and SQL segmentation identifying demographic and behavioral risk factors across 10,000 retail banking customers.
2. **Machine Learning Experiments**: Detailed exploration of supervised models and unsupervised customer segmentation.
3. **Production ML Pipeline & REST API**: A leak-free Scikit-learn `Pipeline` tuned for **ROC-AUC ($0.8603$)**, with calibrated decision thresholding ($0.35$) prioritizing recall ($87.5\%$), served via FastAPI and containerized with Docker.

---

## 🗺️ Repository Map

```
bank-churn-analytics-and-prediction/
├── analytics/                 # 1. Business Intelligence & Analytics
│   ├── power_bi/              # Interactive Power BI dashboard (Bank_Churn_Dashboard.pbix)
│   ├── sql_queries/           # Numbered SQL queries (01_churn_metrics_and_segmentation.sql)
│   └── visualizations/        # Exploratory charts (geography, age, tenure, activity, products)
├── learning_notes/            # 2. Machine Learning Exploration Notes
│   ├── 01_supervised_models_and_tuning.ipynb         # LR baseline, Trees, Random Forest, XGBoost
│   ├── 02_unsupervised_clustering_and_segmentation.ipynb  # K-Means, PCA, Hierarchical, DBSCAN
│   └── 03_deep_learning_ann_scratchpad.ipynb         # Deep learning scratchpad
├── notebooks/                 # Exploratory data analysis & data cleaning
│   └── 01_eda_and_segment_analysis.ipynb             # Full EDA notebook exporting cleaned data
├── data/                      # Dataset repository (churn_raw.csv, churn_cleaned.csv)
├── src/churn/                 # 3. Production Machine Learning Codebase
│   ├── config.py              # Schemas, feature definitions, paths, hyperparameters
│   ├── data.py                # Data loading and stratified train/test splitting
│   ├── pipeline.py            # ColumnTransformer & RandomForest Pipeline
│   ├── train.py               # 5-Fold Stratified CV, GridSearchCV, model training
│   └── evaluate.py            # ROC-AUC, PR-AUC, confusion matrix, threshold tuning
├── models/                    # Serialized model artifacts
│   ├── churn_rf_pipeline.joblib   # Single fitted end-to-end Pipeline
│   └── metadata.json              # Complete training run metadata, metrics, feature importances
├── api/                       # Production FastAPI Service
│   ├── main.py                # Endpoints (GET /health, GET /model-info, POST /predict, POST /predict/batch)
│   └── schemas.py             # Pydantic v2 validation models with strict ranges & enums
├── tests/                     # Automated Test Suite (pytest)
│   ├── test_pipeline.py       # Pipeline, transformer, and data leakage tests
│   └── test_api.py            # API endpoint, boundary validation, and batch inference tests
├── reports/figures/           # High-resolution model evaluation figures
│   ├── roc_curve.png          # ROC curve (AUC = 0.860)
│   ├── precision_recall_curve.png # PR curve (AUC = 0.692)
│   ├── confusion_matrix.png   # Confusion matrix at calibrated threshold 0.35
│   └── feature_importance.png # MDI feature importance ranking
├── docs/                      # Documentation
│   └── interview_notes.md     # 10 senior ML engineering interview defense questions & answers
├── Dockerfile                 # Multi-stage, non-root user, slim base production image
├── .dockerignore              # Clean container build boundaries
├── requirements.txt           # Pinned production and test dependencies
└── README.md                  # Project overview and run guide
```

---

## 📊 Model Performance & Results

Model selection was conducted using **Stratified 5-Fold Cross-Validation** optimizing for **ROC-AUC** on an 80/20 stratified split.

### Model Comparison (Held-Out Test Set: 2,000 customers)

| Model Architecture | Optimization Metric | Test ROC-AUC | Test PR-AUC | Recall | Precision | F1-Score |
|---|---|---|---|---|---|---|
| **Baseline (Logistic Regression)** | Class-weighted loss | $0.7772$ | $0.4679$ | $70.02\%$ | $38.72\%$ | $0.4987$ |
| **Random Forest (Default Thresh $0.50$)** | Stratified 5-Fold CV | **$0.8603$** | **$0.6922$** | $72.48\%$ | $51.48\%$ | **$0.6020$** |
| **Random Forest (Calibrated Thresh $0.35$)** | Recall-Tuned for Retention | **$0.8603$** | **$0.6922$** | **$87.47\%$** | $36.66\%$ | $0.5167$ |

* **Key Takeaway**: At the calibrated threshold of $0.35$, the production model captures **$356$ out of $407$ actual churners** in the test set ($87.5\%$ recall), missing only 51 leaving accounts compared to 112 missed under default $0.50$ thresholding.

### Top Predictive Features (MDI Feature Importance)
1. **Age ($36.4\%$)**: Primary non-linear attrition factor; churn risk peaks heavily in the 45–60 cohort.
2. **NumOfProducts ($23.8\%$)**: Customers holding 3 or 4 products exhibit disproportionately high attrition.
3. **Balance ($9.5\%$)**: High-balance accounts demonstrate acute vulnerability, especially when disengaged.
4. **IsActiveMember ($6.7\%$)**: Active members churn at nearly half the frequency of inactive accounts.
5. **Geography: Germany ($5.4\%$)**: Regional churn density is double that of France or Spain.

---

## ⚙️ Engineering & Design Decisions

* **Why an end-to-end `Pipeline` & `ColumnTransformer`?**
  Prevents **data leakage**. Scaling and imputation parameters are fitted strictly on training folds and automatically applied during inference. Raw customer records can be passed directly to the pipeline without duplicate preprocessing code.
* **Why ROC-AUC instead of Accuracy?**
  With ~20% churn prevalence, accuracy rewards trivial majority-class classifiers. ROC-AUC evaluates true positive vs. false positive ranking across all decision thresholds independently of class balance.
* **Why calibrate decision threshold to 0.35?**
  In retail banking, **false negatives** (losing account balances and customer lifetime value) cost significantly more than **false positives** (a targeted retention email or discounted rate). Lowering the threshold to $0.35$ raises recall to **$87.5\%$** while maintaining defensible precision.
* **Why FastAPI with async `lifespan`?**
  The model pipeline is loaded **once at server startup** into memory (`app.state`), ensuring ultra-low inference latency ($<5\text{ ms}$) with zero per-request disk reads.

---

## 🚀 Quickstart: Running Locally

### 1. Prerequisites & Virtual Environment
```bash
# Clone the repository
git clone https://github.com/Vaidik-Parsai/Bank_data_churn_analysis.git
cd Bank_data_churn_analysis

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Retrain the Pipeline (Optional)
```bash
# Runs Stratified GridSearchCV, threshold tuning, and regenerates plots & metadata
python src/churn/train.py
```

### 3. Run the Automated Test Suite
```bash
# Runs 12 automated unit and integration tests
pytest tests/ -v
```

### 4. Start the FastAPI Prediction Service
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** in your browser to view the interactive OpenAPI / Swagger UI.

---

## 🐳 Running with Docker

### Build the Docker Image
```bash
docker build -t bank-churn-api:latest .
```

### Run the Container
```bash
docker run -d -p 8000:8000 --name bank-churn-service bank-churn-api:latest
```

### Verify Container Health
```bash
docker ps
curl http://localhost:8000/health
```

---

## 📡 API Usage & Example Requests

### 1. Health Check (`GET /health`)
```bash
curl -X GET http://localhost:8000/health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

### 2. Model Information & Metadata (`GET /model-info`)
```bash
curl -X GET http://localhost:8000/model-info
```

### 3. Predict Single Customer Churn (`POST /predict`)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CustomerId": 15634602,
    "CreditScore": 650,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 4,
    "Balance": 85000.50,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 72500.00
  }'
```
**Response:**
```json
{
  "customer_id": 15634602,
  "churn_probability": 0.6421,
  "churn_prediction": true,
  "risk_band": "High",
  "threshold_used": 0.35
}
```

### 4. Batch Prediction (`POST /predict/batch`)
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "CustomerId": 15634602,
        "CreditScore": 650,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 4,
        "Balance": 85000.50,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 72500.00
      },
      {
        "CustomerId": 15634603,
        "CreditScore": 720,
        "Geography": "France",
        "Gender": "Male",
        "Age": 28,
        "Tenure": 6,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 48000.00
      }
    ]
  }'
```
**Response:**
```json
{
  "total_customers": 2,
  "churn_predicted_count": 1,
  "churn_rate_predicted": 0.50,
  "predictions": [
    {
      "customer_id": 15634602,
      "churn_probability": 0.6421,
      "churn_prediction": true,
      "risk_band": "High",
      "threshold_used": 0.35
    },
    {
      "customer_id": 15634603,
      "churn_probability": 0.1142,
      "churn_prediction": false,
      "risk_band": "Low",
      "threshold_used": 0.35
    }
  ]
}
```

---

## 🎯 Interview Walkthrough Guide

For a rapid 5-minute technical interview walkthrough or in-depth technical defense, consult:
- **`docs/interview_notes.md`**: 10 rigorous senior ML interview questions covering leakage prevention, threshold calibration, drift monitoring, latency trade-offs, and stakeholder navigation.

---

### Author
**Vaidik Parsai**  
*Data Analytics • Machine Learning • Production ML Engineering*
