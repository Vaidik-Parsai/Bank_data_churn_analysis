# 🏦 Bank Customer Churn Analysis & Automation Pipeline

## 📌 Project Overview
Customer retention is one of the primary growth drivers in retail banking. This project delivers an end-to-end Business Intelligence solution that analyzes customer demographics, account balances, and product engagement behaviors to identify high-risk segments. 

Using this multi-page interactive dashboard, stakeholders can monitor real-time churn metrics, drill down into behavioral risk patterns, and review algorithmic data-driven recommendations to minimize customer attrition.

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
