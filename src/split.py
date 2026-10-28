import pandas as pd


def make_masks(df: pd.DataFrame, train_q: float = 0.70, val_q: float = 0.85):

    train_step = int(df["step"].quantile(train_q))
    val_step = int(df["step"].quantile(val_q))

    train_mask = df["step"] <= train_step
    val_mask = (df["step"] > train_step) & (df["step"] <= val_step)
    test_mask = df["step"] > val_step
    return train_mask, val_mask, test_mask