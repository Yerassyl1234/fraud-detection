import lightgbm as lgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TRANSACTION_TYPES = ["CASH_OUT", "TRANSFER"]


def make_logreg(continuous, binary, categorical) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(), continuous),
            ("binary", "passthrough", binary),
            ("categorical",
             OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
             categorical),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )),
    ])


def prepare_lgbm_frame(df: pd.DataFrame, columns) -> pd.DataFrame:
    out = df[list(columns)].copy()
    out["transaction_type"] = out["transaction_type"].astype(
        pd.CategoricalDtype(categories=TRANSACTION_TYPES)
    )
    return out


def make_lgbm(**overrides) -> lgb.LGBMClassifier:
    params = dict(
        n_estimators=600,
        learning_rate=0.02,
        num_leaves=15,
        min_child_samples=500,
        scale_pos_weight=1,
        metric="None",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    params.update(overrides)
    return lgb.LGBMClassifier(**params)