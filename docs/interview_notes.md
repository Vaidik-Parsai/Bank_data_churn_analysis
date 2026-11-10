# Senior ML Engineer Interview Defense Notes

This document provides concise, defensible answers to 10 high-probability machine learning engineering interview questions about this repository. Every response is grounded directly in the codebase and architectural choices.

---

### Q1. Why did you wrap your transformations in an sklearn `Pipeline` and `ColumnTransformer` instead of preprocessing upfront in pandas?
**Answer:**
Preprocessing directly in pandas before splitting is the single most common cause of **data leakage** in applied ML. If scaling parameters (mean/variance) or imputation values (median/mode) are computed across the entire dataset, evaluation metrics become overly optimistic because the test set information has contaminated the training process. 

In this codebase (`src/churn/pipeline.py`), `ColumnTransformer` encapsulates:
- Numeric imputation (`SimpleImputer(strategy="median")`) and standardization (`StandardScaler`)
- Categorical imputation (`SimpleImputer(strategy="most_frequent")`) and one-hot encoding (`OneHotEncoder(handle_unknown="ignore")`)

By coupling preprocessing with the estimator in a unified `Pipeline`, `fit()` is executed strictly on the training partition (`X_train`), and identical transformations are mechanically guaranteed during cross-validation, test evaluation, and production serving. Furthermore, it allows serving raw, un-transformed JSON payloads directly to the model without maintaining parallel preprocessing code in production.

---

### Q2. How is class imbalance handled, and why didn't you select the model using Accuracy?
**Answer:**
The bank churn dataset exhibits an imbalanced class distribution: **20.37% churners (Class 1) vs 79.63% retained customers (Class 0)**. 

- **Why Accuracy is Deceptive:** A trivial majority-class classifier predicting 0 for every customer achieves nearly 80% accuracy while failing to identify a single churned customer (0% recall).
- **Our Handling Strategy:** 
  1. We initialized estimators with `class_weight="balanced"` (`src/churn/pipeline.py`), which automatically adjusts loss weights inversely proportional to class frequencies ($w_j = \frac{N}{2 \cdot N_j}$).
  2. We configured `GridSearchCV` to optimize **ROC-AUC** as the primary selection criterion (`src/churn/train.py`, line 96) and evaluated **PR-AUC (Average Precision)**.
  3. Stratified splits (`StratifiedKFold` and `stratify=y` in `train_test_split`) ensure consistent class ratios across all validation folds.

---

### Q3. How did you choose the decision threshold of 0.35, and what is the underlying business cost function?
**Answer:**
Default classification thresholds ($p = 0.50$) implicitly assume symmetric misclassification costs:
$$\text{Cost}(\text{False Positive}) = \text{Cost}(\text{False Negative})$$
In banking retention, this assumption is false:
- **False Negative (Missed Churner):** The bank loses customer deposits, recurring revenue, and incurs replacement acquisition costs (\$500–\$2,000+ per account).
- **False Positive (False Alarm):** The bank sends a retention email, promotional rate, or customer care check-in (\$5–\$25 intervention cost).

In `src/churn/evaluate.py`, we evaluated threshold candidates from $0.20$ to $0.70$:
- At default **0.50**: Recall is **72.48%** (112 churners missed).
- At tuned **0.35**: Recall increases to **87.47%** (only 51 churners missed; 356 out of 407 churners caught), while maintaining an acceptable precision of **36.66%**.

Lowering the threshold to 0.35 provides optimal coverage for the bank's proactive customer success outreach without overwhelming retention resources.

---

### Q4. Why did you select Random Forest over a linear model or deep learning?
**Answer:**
1. **Vs. Baseline (Logistic Regression):** As demonstrated in `models/metadata.json`, Logistic Regression achieved a cross-validation ROC-AUC of **0.7662**, whereas tuned Random Forest achieved **0.8570** (Test ROC-AUC: **0.8603**). Tabular banking data contains non-linear interactions (e.g. balance vs. active membership, tenure vs. product count) that tree ensembles capture without manual interaction engineering.
2. **Vs. Deep Learning:** On tabular datasets of $10,000$ rows, deep neural networks are computationally heavy, prone to overfitting, lack tree-based interpretability, and require extensive tuning of embeddings and learning rate schedules. Random Forest provides superior sample efficiency, deterministic convergence, out-of-the-box feature importances, and low-latency inference ($<5\text{ ms}$).

---

### Q5. How does the pipeline handle missing values or unexpected categorical values at inference time?
**Answer:**
Production data inevitably encounters missing fields or out-of-vocabulary inputs. Our pipeline builds defense directly into the transformers (`src/churn/pipeline.py`):
- **Missing Values:** Numerical fields are median-imputed, and categorical fields are mode-imputed using parameters learned during training.
- **Unseen Categories:** `OneHotEncoder` is initialized with `handle_unknown="ignore"`. If an unrecognized country (e.g. `"Canada"`) or gender value arrives, the encoder generates an all-zero vector for those category columns rather than raising a runtime exception.
- **Input Validation:** At the network boundary (`api/schemas.py`), Pydantic strictly validates types, bounds, and enums before requests ever reach the ML model, returning clean HTTP 422 errors for malformed payloads.

---

### Q6. How is the model loaded in the FastAPI service, and how would you scale this to high traffic?
**Answer:**
- **Current Architecture:** In `api/main.py`, the pipeline and metadata are loaded **once during application startup** inside the async `lifespan` context manager into `app.state`. Incoming requests reuse the in-memory pipeline, avoiding disk I/O and deserialization latency per request.
- **Horizontal Scaling:** Because the FastAPI service is stateless, it scales horizontally across multiple container replicas behind a load balancer (e.g. AWS ALB, NGINX, or Kubernetes Ingress).
- **Concurrency:** CPU-bound model inference is served with multiple Uvicorn workers (`uvicorn --workers 4`), while asynchronous endpoints handle non-blocking health checks and telemetry.

---

### Q7. What are the top drivers of churn in this model, and do they align with your exploratory data analysis?
**Answer:**
Extracted from `models/metadata.json` via MDI (Mean Decrease in Impurity):
1. **Age (36.4% importance):** Strongest non-linear predictor. Customers aged 45–60 exhibit disproportionately high churn rates.
2. **NumOfProducts (23.8% importance):** Customers with 3 or 4 bank products show extreme churn propensity (counter-intuitive, reflecting fee structures or product bundling friction).
3. **Balance (9.5% importance):** High-balance accounts demonstrate specific vulnerability, particularly when inactive.
4. **IsActiveMember (6.7% importance):** Engagement is a primary retention lever; active members churn at half the rate of inactive members.
5. **Geography_Germany (5.4% importance):** German customers exhibit a ~32% churn rate (double France and Spain).

These model drivers perfectly validate the empirical findings visualised in `analytics/visualizations/` and the Power BI dashboard (`analytics/power_bi/Bank_Churn_Dashboard.pbix`).

---

### Q8. How do you prevent identifier leakage during model training and production serving?
**Answer:**
Identifiers such as `RowNumber`, `CustomerId`, and `Surname` carry high cardinality and spurious correlations that cause severe overfitting.
In `src/churn/config.py`:
```python
IDENTIFIER_COLS = ["RowNumber", "CustomerId", "Surname"]
```
In `src/churn/data.py`:
`prepare_features()` explicitly drops these columns from the feature matrix if present. In the API (`api/schemas.py`), `CustomerId` is optional and captured strictly for audit logging/response tagging, but excluded from the DataFrame passed to `model.predict_proba()`.

---

### Q9. How would you monitor this model for data drift and concept drift in production?
**Answer:**
1. **Data Drift (Feature Distribution Shift):** 
   - Monitor numerical features (e.g., `Age`, `Balance`, `EstimatedSalary`) using two-sample Kolmogorov-Smirnov (KS) tests or Population Stability Index (PSI). A $\text{PSI} > 0.25$ indicates significant distributional drift.
   - Monitor categorical distributions (residence countries, gender ratios) using Chi-square contingency tests.
2. **Concept Drift / Covariate Shift ($P(Y|X)$):**
   - Track monthly predicted churn rate vs. actual customer attrition (observed 30–90 days post-prediction).
   - If macro-economic shifts (e.g. interest rate adjustments) alter customer behavior, trigger an automated retraining pipeline on recent trailing-window data.

---

### Q10. If an executive demands: "We need 90% precision to stop wasting retention budget on false alarms," how do you respond?
**Answer:**
I would present the empirical **Precision-Recall Curve** (`reports/figures/precision_recall_curve.png`) and the **Threshold Selection Grid** (`models/metadata.json`):
1. **Trade-off Quantification:** On this dataset, reaching $75\%$ precision requires increasing the decision threshold to $0.70$, which collapses recall from $87.5\%$ down to **$47.4\%$**—meaning the bank would miss more than half of all leaving customers.
2. **Financial Cost Framing:** I would model the business cost function:
   $$\text{Total Cost} = (\text{FN} \times \text{LTV Lost}) + (\text{FP} \times \text{Intervention Cost})$$
   Because a lost customer costs orders of magnitude more than an email or promotional offer, maximizing precision at the expense of recall causes higher net financial losses.
3. **Alternative Solution:** Instead of raising the threshold globally, implement **tiered interventions**:
   - `High Risk` ($p \ge 0.60$): Direct phone outreach from account managers (higher cost).
   - `Medium Risk` ($0.35 \le p < 0.60$): Automated email engagement and automated satisfaction survey (near-zero cost).
   - `Low Risk` ($p < 0.35$): Standard banking communications.
