CREATE TABLE transactions_raw (
    step                 NUMBER(4),
    transaction_type     VARCHAR2(20),
    amount               NUMBER(18, 2),
    name_orig            VARCHAR2(30),
    old_balance_orig     NUMBER(18, 2),
    new_balance_orig     NUMBER(18, 2),
    name_dest            VARCHAR2(30),
    old_balance_dest     NUMBER(18, 2),
    new_balance_dest     NUMBER(18, 2),
    is_fraud             NUMBER(1),
    is_flagged_fraud     NUMBER(1)
);
SELECT transaction_type, COUNT(*) as transactions_count, SUM(is_fraud) as fraud_count
FROM transactions_raw
GROUP BY transaction_type
ORDER BY fraud_count DESC;

CREATE OR REPLACE VIEW transactions_risk AS
SELECT *
FROM transactions_raw
WHERE transaction_type IN ('CASH_OUT', 'TRANSFER');

SELECT COUNT(*)
FROM transactions_risk;

SELECT 
    transaction_type, 
    COUNT(*) AS transactions_count, 
    SUM(is_fraud) AS fraud_count,
    ROUND(
        SUM(is_fraud) / COUNT(*) * 100,
        4
    ) AS fraud_rate_percent
FROM transactions_risk
GROUP BY transaction_type
ORDER BY fraud_rate_percent DESC;


SELECT
    transaction_type,
    is_fraud,
    COUNT(*) AS transactions_count,
    ROUND(AVG(amount), 2) AS avg_amount,
    ROUND(MEDIAN(amount), 2) AS median_amount,
    ROUND(MAX(amount), 2) AS max_amount
FROM transactions_risk
GROUP BY
    transaction_type,
    is_fraud
ORDER BY
    transaction_type,
    is_fraud;





