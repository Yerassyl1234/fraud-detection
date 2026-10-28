import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score

from src.evaluation import precision_at_k
from src.features import FINAL_FEATURES, add_features
from src.models import make_lgbm, prepare_lgbm_frame
from src.split import make_masks


def main(data_path: str, model_path: str) -> None:
    df = add_features(pd.read_parquet(data_path))
    train_mask, val_mask, test_mask = make_masks(df)

    X_train = prepare_lgbm_frame(df.loc[train_mask], FINAL_FEATURES)
    y_train = df.loc[train_mask, "is_fraud"]

    model = make_lgbm()
    model.fit(X_train, y_train)

    for name, mask in [("validation", val_mask), ("test", test_mask)]:
        X = prepare_lgbm_frame(df.loc[mask], FINAL_FEATURES)
        y = df.loc[mask, "is_fraud"]
        proba = model.predict_proba(X)[:, 1]
        print(f"PR-AUC {name}: {average_precision_score(y, proba):.3f}")
        if name == "test":
            print(precision_at_k(y, proba).to_string(index=False))

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": FINAL_FEATURES}, model_path)
    print("Saved:", model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/model_df.parquet")
    parser.add_argument("--out", default="models/lgbm_fraud.joblib")
    args = parser.parse_args()
    main(args.data, args.out)