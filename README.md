# 🏦 Bank Customer Churn Analysis

An end-to-end data analytics project focused on identifying the factors associated with customer churn and translating those findings into actionable customer-retention strategies.

The project combines **Python, SQL, and Power BI** to move from raw customer data to business insights and an interactive dashboard.

---

## 📊 Project Overview

Customer churn is an important challenge for banks because losing existing customers can reduce customer lifetime value and increase acquisition costs.

This project analyzes **10,000 bank customers** to answer questions such as:

* Which customer segments have the highest churn?
* Does geography influence customer retention?
* Are inactive customers more likely to leave?
* Which age groups are most at risk?
* Does the number of products held by a customer affect churn?
* How does customer tenure relate to retention?
* Which customer groups should receive priority in retention campaigns?

---

## 🎯 Key Metrics

| Metric                  |             Result |
| ----------------------- | -----------------: |
| Total Customers         |             10,000 |
| Churned Customers       |              2,037 |
| Overall Churn Rate      |             20.37% |
| Highest-Churn Geography |   Germany (~32.4%) |
| Highest-Risk Age Group  |              51–60 |
| Major Engagement Risk   | Inactive Customers |

Approximately **1 in every 5 customers** in the dataset has exited the bank.

---

## 📈 Dashboard

The Power BI dashboard provides an executive-level view of customer churn and allows churn patterns to be explored across customer segments.

### Dashboard Preview

> Add your exported Power BI dashboard screenshot here.

```markdown
![Bank Churn Dashboard](visualizations/dashboard_overview.png)
```

The dashboard focuses on:

* Overall customer churn
* Geographic churn patterns
* Customer age groups
* Active vs inactive customers
* Product ownership
* Customer tenure
* Customer demographics

---

## 🔍 Key Findings

### 🌍 1. Geography Matters

Germany has a churn rate of approximately **32.4%**, substantially higher than France and Spain.

This makes German customers an important segment for deeper investigation and targeted retention efforts.

---

### 👤 2. Inactive Customers Are at Greater Risk

Inactive customers show a substantially higher churn rate than active customers.

This indicates that **customer engagement is one of the strongest retention signals** observed in the dataset.

Customers showing declining activity could therefore be targeted before they completely disengage from the bank.

---

### 🎂 3. Middle-Aged Customers Show Higher Churn

The **51–60 age group** records the highest churn rate, followed by customers aged **41–50**.

This suggests that retention strategies may need to consider differences in customer needs across age groups rather than applying the same approach to every customer.

---

### 💳 4. Product Ownership Reveals an Important Pattern

Customers holding **3 or 4 banking products** show unusually high churn rates.

Although these groups contain fewer customers, the result suggests that greater product ownership does not automatically translate into stronger retention.

Possible factors worth investigating include:

* Product complexity
* Fees
* Bundling issues
* Cross-selling practices
* Customer service experience

---

### ⏳ 5. Tenure Has a Smaller Relationship With Churn

Churn decreases slightly as customer tenure increases.

However, the difference is considerably smaller than the differences observed for geography, age, and customer activity.

This suggests that tenure alone should not be used to identify high-risk customers.

---

## 🚨 High-Risk Customer Segmentation

Instead of analyzing each customer characteristic independently, the project also examines combinations of high-risk characteristics.

One important segment consists of customers who are:

**German + Inactive + Middle-aged**

Combining multiple behavioural and demographic indicators provides a more useful basis for retention prioritization than relying on a single variable.

---

## 💡 Business Recommendations

### 1. Prioritize Retention Efforts in Germany

Investigate why German customers churn at a considerably higher rate.

Possible areas for further investigation include:

* Regional pricing
* Competitor activity
* Customer service quality
* Product suitability
* Customer experience

### 2. Re-engage Inactive Customers

Customer inactivity is strongly associated with churn.

The bank could use:

* Personalized engagement campaigns
* Behaviour-triggered communication
* Product usage incentives
* Proactive customer support

to re-engage customers before they leave.

### 3. Focus on High-Risk Age Segments

Customers aged **41–60**, particularly the **51–60 segment**, should receive additional attention when combined with other churn indicators.

### 4. Investigate Multi-Product Customer Churn

The unusually high churn among customers holding three or four products should be investigated for possible product or service friction.

### 5. Use Multi-Factor Customer Segmentation

Retention decisions should combine factors such as:

**Geography + Activity + Age + Product Usage + Financial Behaviour**

rather than relying on a single churn indicator.

---

## 🛠️ Tools & Technologies

| Technology       | Purpose                                      |
| ---------------- | -------------------------------------------- |
| Python           | Data cleaning and exploratory analysis       |
| Pandas           | Data manipulation and aggregation            |
| NumPy            | Numerical operations                         |
| Matplotlib       | Data visualization                           |
| SQL              | Business analysis and customer segmentation  |
| Power BI         | Interactive dashboard and business reporting |
| Jupyter Notebook | Analytical workflow                          |

---

## 🔄 Project Workflow

```text
Raw Customer Data
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Exploratory Data Analysis
        ↓
Customer Segmentation
        ↓
SQL Analysis
        ↓
Power BI Dashboard
        ↓
Business Insights
        ↓
Retention Recommendations
```

---

## 📁 Repository Structure

```text
Bank_data_churn_analysis/
│
├── data/
│   ├── churn_raw.csv
│   └── churn_cleaned.csv
│
├── notebooks/
│   └── 01_bank_churn_analysis.ipynb
│
├── power_bi/
│   └── Bank_Churn_Dashboard.pbix
│
├── sql_queries/
│   └── bank_churn_analysis.sql
│
├── visualizations/
│   ├── churn_by_activity.png
│   ├── churn_by_age_group.png
│   ├── churn_by_balance.png
│   ├── churn_by_gender.png
│   ├── churn_by_geography.png
│   ├── churn_by_products.png
│   └── churn_by_tenure.png
│
├── .gitignore
└── README.md
```

---

## 🧹 Data Preparation

The raw dataset was inspected for:

* Missing values
* Duplicate records
* Data types
* Irrelevant identifier columns

Identifier fields that do not contribute to customer-behaviour analysis were removed.

Additional features were created for analysis, including:

* Age groups
* Balance categories
* Tenure groups

The cleaned dataset is stored separately to maintain a clear distinction between **raw and processed data**.

---

## 📊 Analysis Approach

For categorical variables, churn rate is calculated using the mean of the `Exited` variable:

```text
Exited = 0 → Customer Retained
Exited = 1 → Customer Churned
```

Therefore:

```text
Average of Exited × 100 = Churn Rate (%)
```

This allows churn rates to be compared fairly between customer segments of different sizes.

---

## ⚠️ Limitations

This analysis has several limitations:

* The dataset represents a static customer snapshot rather than longitudinal customer behaviour.
* Transaction-level banking activity is unavailable.
* Customer satisfaction and service interaction data are unavailable.
* Competitor offers and regional market conditions are unknown.
* Relationships identified during analysis should not automatically be interpreted as causal.
* Detailed profitability information would be required to estimate the financial impact of retention strategies.

---

## 🚀 Future Improvements

Potential extensions include:

* Customer churn probability modelling
* Customer-level risk scoring
* Automated high-risk customer identification
* More detailed customer lifetime value analysis
* Transaction-level behavioural features
* Retention campaign effectiveness analysis

Machine-learning experimentation can be developed separately using the cleaned dataset while keeping this repository focused on the core **data analytics and business intelligence workflow**.

---

## 📌 Conclusion

The analysis shows that customer churn is not distributed evenly across the bank's customer base.

**Geography, customer activity, age, and product ownership** emerge as important factors associated with churn.

Rather than targeting every customer equally, the bank could use these insights to prioritize high-risk customer segments and develop more focused retention strategies.

---

### Author

**Vaidik Parsai**

Data Analytics • Data Science • Machine Learning
