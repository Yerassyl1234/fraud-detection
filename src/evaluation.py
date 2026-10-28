import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# thresholds from 0.50 up to very close to 1 (boosted models are often
# extremely confident, so a grid that stops at 0.999 can be too short)
DEFAULT_THRESHOLDS = np.concatenate([
    np.arange(0.50, 0.99, 0.01),
    1 - np.logspace(-2, -6, 25),
])


def metrics_at_threshold(y_true, proba, threshold: float) -> dict:

    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def evaluate(model, X, y_true, threshold: float) -> dict:
    proba = model.predict_proba(X)[:, 1]
    return metrics_at_threshold(y_true, proba, threshold)


def pick_threshold(proba, y_true, min_recall: float = 0.80,
                   thresholds=DEFAULT_THRESHOLDS) -> float:
    best_threshold, best_precision = None, -1.0
    for t in thresholds:
        pred = (np.asarray(proba) >= t).astype(int)
        if recall_score(y_true, pred) >= min_recall:
            prec = precision_score(y_true, pred, zero_division=0)
            if prec > best_precision:
                best_threshold, best_precision = float(t), prec

    if best_threshold is None:
        print(f"Warning: recall {min_recall:.0%} is not reachable, using 0.5")
        return 0.5
    return best_threshold


def precision_at_k(y_true, proba, ks=(500, 1000, 2000, 3000, 4000, 5000, 8000)) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(proba), kind="stable")
    y_sorted = y_true[order]
    total_fraud = int(y_true.sum())

    rows = []
    for k in ks:
        tp = int(y_sorted[:k].sum())
        rows.append({
            "k": k,
            "caught": tp,
            "precision@k": round(tp / k, 3),
            "recall@k": round(tp / total_fraud, 3),
        })
    return pd.DataFrame(rows)