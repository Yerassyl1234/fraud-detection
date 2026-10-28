OPTIONS (SKIP=1, DIRECT=TRUE, ERRORS=1000)

LOAD DATA
INFILE 'data/raw/paysim.csv'
BADFILE 'data/processed/paysim.bad'
DISCARDFILE 'data/processed/paysim.discard'

INTO TABLE transactions_raw

FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
    step                 INTEGER EXTERNAL,
    transaction_type     CHAR,
    amount               DECIMAL EXTERNAL,
    name_orig            CHAR,
    old_balance_orig     DECIMAL EXTERNAL,
    new_balance_orig     DECIMAL EXTERNAL,
    name_dest            CHAR,
    old_balance_dest     DECIMAL EXTERNAL,
    new_balance_dest     DECIMAL EXTERNAL,
    is_fraud             INTEGER EXTERNAL,
    is_flagged_fraud     INTEGER EXTERNAL
)