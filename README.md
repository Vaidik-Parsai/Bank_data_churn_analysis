# 🏦 Bank Customer Churn Analysis & Automation Pipeline

## 📌 Project Overview
Customer retention is one of the primary growth drivers in retail banking. This repository brings together an end-to-end customer churn initiative across three distinct pillars: Business Intelligence / Analytics, Machine Learning Exploration, and a Production-Grade ML Serving Service.

---

## 🗺️ Repository Map

```
bank-churn-analytics-and-prediction/
├── analytics/                 # 1. Data-Analyst Deliverables
│   ├── power_bi/              # Interactive Power BI dashboard (.pbix)
│   ├── sql_queries/           # Production SQL analysis and segmentation queries
│   └── visualizations/        # Exploratory charts and visual distributions
├── learning_notes/            # 2. Machine Learning Exploration Notes
│   ├── 01_supervised_models_and_tuning.ipynb         # LR baseline, Trees, Random Forest, XGBoost
│   ├── 02_unsupervised_clustering_and_segmentation.ipynb  # K-Means, PCA, Hierarchical, DBSCAN
│   └── 03_deep_learning_ann_scratchpad.ipynb         # Deep learning scratchpad
├── notebooks/                 # EDA & BI data preparation
│   └── 01_eda_and_segment_analysis.ipynb             # Full exploratory data analysis & cleaning
├── data/                      # Dataset repository (raw & cleaned CSVs)
├── src/churn/                 # 3. Production ML Codebase (Scikit-Learn Pipeline)
│   ├── config.py              # Schema, feature groups, random states
│   ├── data.py                # Data loading and stratified splitting
│   ├── pipeline.py            # ColumnTransformer & RandomForest Pipeline
│   ├── train.py               # Cross-validation & hyperparameter tuning
│   └── evaluate.py            # Metrics, confusion matrix, ROC-AUC evaluation
├── models/                    # Serialized pipeline artifact & metadata.json
├── api/                       # Production FastAPI prediction service
├── tests/                     # Unit & integration tests (pipeline & API)
├── reports/figures/           # Evaluation charts (ROC curve, confusion matrix, importance)
├── docs/                      # Technical documentation & interview defense notes
└── Dockerfile, requirements.txt, .gitignore, README.md
```

### Three Parts Explained
1. **Analytics & Business Intelligence (`analytics/` & `notebooks/`)**: Interactive Power BI dashboard, analytical SQL queries (demographics, segmentation, cohort retention), and exploratory data analysis.
2. **Machine Learning Notes (`learning_notes/`)**: Systematic explorations of supervised classification algorithms (Logistic Regression, Decision Trees, Random Forest, XGBoost) and unsupervised customer segmentation (K-Means, PCA, Hierarchical, DBSCAN).
3. **Production ML & API Service (`src/churn/`, `api/`, `models/`)**: An interview-ready, production-grade pipeline built with scikit-learn, tuned for ROC-AUC on imbalanced churn data, packaged with serialized metadata, and deployed via a validated FastAPI REST API.

---

## 🛠️ Tech Stack & Architecture
* **Data Source:** Bank Churn Customer Profile Dataset (`churn.csv`)
* **BI Platform:** Power BI Desktop
* **Data Modeling & Analytics:** DAX (Data Analysis Expressions)
* **Version Control:** Git & GitHub

---

## 📊 Dashboard Architecture & Key Visuals

### 1. Executive Summary
* **Core Metrics Tracked:** Total Customers (10K), Total Churned Customers (2K), Baseline Churn Rate (20.37%), Average Customer Age (38.92), and Active Membership volume.
* **Purpose:** High-level overview for senior stakeholders to assess overall organizational health at a glance.

### 2. Customer Demographics
* **Age Group Segmentation:** Custom-engineered DAX groupings (`18-30`, `31-40`, `41-50`, `51-60`, `60+`) mapping the highest churn propensity to the 51-60 age category.
* **Geographic Mapping:** Highlighting regional risk allocation across France, Spain, and Germany.
* **Gender Analysis:** A unified split highlighting variations in churn behaviors between male and female cohorts.

### 3. Behavioral Analysis
* **Tenure Evaluation:** Tracking customer lifespan patterns, showing an increased risk peak right around Year 1.
* **Product Engagement:** Cross-referencing account vulnerability with the total number of banking products utilized.

### 4. Advanced Insights & DAX Formulations
This page houses programmatic business rules to dynamically flag vulnerable accounts and estimate financial impact:
* **Customer Lifetime Value (CLV Calculation):** Establishes baseline financial value trapped in non-churned assets.
* **At-Risk Customer Classifier:** Programmatic identifier flagging inactive, low-tenure (< 3 years), high-balance (> 50,000) accounts.

---

## 💡 Strategic Business Recommendations
1. **Targeted German Retention Initiative:** Germany exhibits a **32% churn rate**—exactly double the regional average. A targeted localized audit into pricing structures or localized competitor offerings is recommended.
2. **Product Optimization Program:** Customers holding 3 or 4 products display disproportionately high churn rates. Product bundling and account service structures should be re-evaluated for friction points.
3. **Mid-Age Engagement Campaign:** The 51-60 demographic shows prominent churn density. Introducing premium wealth preservation or transition services could capture and lock in this stable segment.
