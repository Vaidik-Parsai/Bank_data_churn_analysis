
/*
========================================================
BANK CUSTOMER CHURN ANALYSIS
========================================================

Purpose:
Analyze customer churn patterns using SQL and identify
high-risk customer segments for retention strategies.

Dataset: Bank Customer Churn
Target: Exited (1 = Churned, 0 = Retained)
========================================================
*/


Select
    count(*) as total_customers,
    sum(case when Exited=1 then 1 else 0 end) as churned_custromers,
    round(sum(case when Exited=1 then 1 else 0 end) * 100.0 / count(*), 2) as churn_rate
from Customers;


Select
    Geography,
    count(*) as total_customers,
    round(sum(case when Exited=1 then 1 else 0 end ) *100.0 / count(*), 2) as churn_rate,
    round(avg(credit_score),0) as avg_credit_score
from Customers
group by Geography
order by churn_rate desc;


WITH customer_segments AS (
    SELECT 
        CustomerId,
        CASE 
            WHEN Balance > 100000 AND IsActiveMember = 1 THEN 'High Value Active'
            WHEN Balance > 100000 AND IsActiveMember = 0 THEN 'High Value Inactive'
            WHEN Balance BETWEEN 50000 AND 100000 THEN 'Medium Value'
            ELSE 'Low Value'
        END as segment,
        Exited
    FROM customers
)
SELECT 
    segment,
    COUNT(*) as total,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2) as churn_rate
FROM customer_segments
GROUP BY segment
ORDER BY churn_rate DESC;


SELECT 
    Tenure,
    COUNT(*) as customers,
    ROUND(100.0 * SUM(Exited) OVER(PARTITION BY Tenure) / COUNT(*) OVER(PARTITION BY Tenure), 2) as churn_rate,
    ROUND(AVG(Balance) OVER(PARTITION BY Tenure), 0) as avg_balance
FROM customers
GROUP BY Tenure
ORDER BY Tenure;


