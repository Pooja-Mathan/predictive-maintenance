"""Train and evaluate failure-prediction models.

Usage:
    python -m src.train
    python -m src.train --data data/ai4i2020.csv --model-out models/model.joblib
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_preprocessing import build_preprocessor, prepare_dataset

load_dotenv()


def get_models(seed: int) -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }


def evaluate(pipe: Pipeline, X_test, y_test) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train predictive maintenance model")
    parser.add_argument("--data", default=os.getenv("DATA_PATH", "data/ai4i2020.csv"))
    parser.add_argument("--model-out", default=os.getenv("MODEL_PATH", "models/model.joblib"))
    parser.add_argument("--metrics-out", default=os.getenv("METRICS_PATH", "models/metrics.json"))
    parser.add_argument("--seed", type=int, default=int(os.getenv("RANDOM_STATE", 42)))
    parser.add_argument("--test-size", type=float, default=float(os.getenv("TEST_SIZE", 0.2)))
    args = parser.parse_args()

    X, y = prepare_dataset(args.data)
    print(f"Loaded {len(X)} rows | failure rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    results, fitted = {}, {}
    for name, clf in get_models(args.seed).items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        results[name] = evaluate(pipe, X_test, y_test)
        fitted[name] = pipe
        r = results[name]
        print(f"{name:22s} F1={r['f1']:.3f} P={r['precision']:.3f} "
              f"R={r['recall']:.3f} ROC-AUC={r['roc_auc']:.3f} PR-AUC={r['pr_auc']:.3f}")

    # Select by PR-AUC: more informative than accuracy for imbalanced data (~3.4% failures)
    best = max(results, key=lambda k: results[k]["pr_auc"])
    print(f"\nBest model: {best}")
    print(classification_report(y_test, fitted[best].predict(X_test),
                                target_names=["No failure", "Failure"], zero_division=0))

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    joblib.dump(fitted[best], args.model_out)
    with open(args.metrics_out, "w") as f:
        json.dump({"best_model": best, "results": results}, f, indent=2)
    print(f"Saved model -> {args.model_out}\nSaved metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()
