"""Score new transactions with the saved model.

Example:
    from src.score import load_model, score_one

    bundle = load_model("models/lgbm_fraud.joblib")
    risk = score_one(bundle, {
        "step": 10,
        "transaction_type": "TRANSFER",
        "amount": 250_000.0,
        "old_balance_orig": 250_000.0,
        "old_balance_dest": 0.0,
    })
    print(risk)

The output is a risk score between 0 and 1 used for ranking transactions.
It is not a calibrated probability of fraud.
"""

import joblib
import pandas as pd

from src.features import add_features
from src.models import prepare_lgbm_frame

DEFAULT_MODEL_PATH = "models/lgbm_fraud.joblib"


def load_model(path: str = DEFAULT_MODEL_PATH) -> dict:
    return joblib.load(path)


def score_transactions(bundle: dict, df: pd.DataFrame) -> pd.Series:
    features = add_features(df)
    X = prepare_lgbm_frame(features, bundle["columns"])
    return pd.Series(bundle["model"].predict_proba(X)[:, 1], index=df.index, name="risk")


def score_one(bundle: dict, transaction: dict) -> float:
    return float(score_transactions(bundle, pd.DataFrame([transaction])).iloc[0])