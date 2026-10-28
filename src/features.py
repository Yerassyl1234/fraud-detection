import pandas as pd 
import numpy as np

FINAL_CONTINUOUS = [
    "log_amount",
    "log_old_balance_orig",
    "log_old_balance_dest",
    "hour_sin",
    "hour_cos",
]
FINAL_BINARY = [
    "origin_zero_balance",
    "destination_zero_balance",
]
CATEGORICAL = ["transaction_type"]

FINAL_FEATURES = FINAL_CONTINUOUS + FINAL_BINARY + CATEGORICAL


LOGREG_CONTINUOUS = FINAL_CONTINUOUS + ["log_amount_to_balance_ratio"]
 
SHORTCUT_FEATURES = ["log_amount_to_balance_ratio", "full_balance_attempt"]
 
REQUIRED_COLUMNS = ["step", "transaction_type", "amount",
                    "old_balance_orig", "old_balance_dest"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
 
    df = df.copy()
 
    # time of day as a point on a circle (23:00 is close to 00:00)
    hour = (df["step"] - 1) % 24
    df["hour"] = hour.astype("int8")
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
 
    # money features on a log scale
    df["log_amount"] = np.log1p(df["amount"]).astype("float32")
    df["log_old_balance_orig"] = np.log1p(df["old_balance_orig"]).astype("float32")
    df["log_old_balance_dest"] = np.log1p(df["old_balance_dest"]).astype("float32")
 
    # empty accounts
    df["origin_zero_balance"] = (df["old_balance_orig"] == 0).astype("int8")
    df["destination_zero_balance"] = (df["old_balance_dest"] == 0).astype("int8")
 
    # shortcut features (analysis only, not used by the final model)
    has_balance = df["old_balance_orig"] > 0
    ratio = np.zeros(len(df), dtype="float64")
    np.divide(
        df["amount"].to_numpy(),
        df["old_balance_orig"].to_numpy(),
        out=ratio,
        where=has_balance.to_numpy(),
    )
    df["log_amount_to_balance_ratio"] = np.log1p(ratio).astype("float32")
    df["full_balance_attempt"] = (
        has_balance
        & np.isclose(df["amount"], df["old_balance_orig"], rtol=0, atol=0.01)
    ).astype("int8")
 
    return df
 
 
def add_recipient_history(df: pd.DataFrame, dest_col: str = "name_dest") -> pd.DataFrame:
    
    df = df.copy()
 
    s = df[[dest_col, "step"]].sort_values([dest_col, "step"], kind="stable")
    position_in_account = s.groupby(dest_col).cumcount()
    position_in_step = s.groupby([dest_col, "step"]).cumcount()
    prev_ops = position_in_account - position_in_step
 
    df["dest_prev_ops"] = prev_ops.reindex(df.index).astype("int32")
    df["log_dest_prev_ops"] = np.log1p(df["dest_prev_ops"]).astype("float32")
    df["dest_is_new"] = (df["dest_prev_ops"] == 0).astype("int8")
    return df
 