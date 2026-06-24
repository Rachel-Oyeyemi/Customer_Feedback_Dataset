"""
train_model.py
===============

This module provides utilities to train machine‑learning models for sentiment
classification.  It supports training a baseline logistic regression model and an
advanced random forest model using precomputed TF‑IDF features and labels.
Models are persisted to disk for later evaluation and inference.

Functions
---------
train_models:
    Train baseline and advanced models on TF‑IDF features and save them to disk.
"""

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_models(
    features_path: str,
    labels_path: str,
    baseline_model_path: str,
    advanced_model_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[str, str]:
    """Train baseline and advanced models using TF‑IDF features.

    Parameters
    ----------
    features_path : str
        Path to the joblib file containing the TF‑IDF feature matrix.
    labels_path : str
        Path to the joblib file containing the labels list.
    baseline_model_path : str
        File path to save the trained logistic regression model.
    advanced_model_path : str
        File path to save the trained random forest model.
    test_size : float, optional
        Proportion of the dataset to include in the test split (default is 0.2).
    random_state : int, optional
        Random seed for reproducibility (default is 42).

    Returns
    -------
    Tuple[str, str]
        Paths to the saved baseline and advanced model files.

    Raises
    ------
    FileNotFoundError
        If the feature or label files cannot be found.
    Exception
        For any other error during model training or saving.
    """
    # Load features and labels
    features_file = Path(features_path)
    labels_file = Path(labels_path)
    if not features_file.exists() or not labels_file.exists():
        raise FileNotFoundError("Feature or label file not found.")

    logging.info("Loading features and labels from disk")
    X = joblib.load(features_file)
    y = joblib.load(labels_file)

    # Split into train and test sets for evaluation during training
    logging.info("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train baseline model: Logistic Regression
    logging.info("Training baseline Logistic Regression model")
    # For multi‑class classification we rely on the default one‑vs‑rest solver.  Some versions
    # of scikit‑learn do not support `multi_class` parameter with certain solvers, so we
    # stick to default settings and allow the library to choose the appropriate solver.
    baseline_clf = LogisticRegression(max_iter=1000)
    baseline_clf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = baseline_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    logging.info(
        "Baseline model performance – Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
        acc,
        precision,
        recall,
        f1,
    )

    # Train advanced model: Random Forest
    logging.info("Training advanced Random Forest model")
    advanced_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )
    advanced_clf.fit(X_train, y_train)

    # Evaluate advanced model
    y_pred_adv = advanced_clf.predict(X_test)
    acc_adv = accuracy_score(y_test, y_pred_adv)
    precision_adv, recall_adv, f1_adv, _ = precision_recall_fscore_support(
        y_test, y_pred_adv, average="weighted"
    )
    logging.info(
        "Advanced model performance – Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
        acc_adv,
        precision_adv,
        recall_adv,
        f1_adv,
    )

    # Ensure output directory exists
    Path(baseline_model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(advanced_model_path).parent.mkdir(parents=True, exist_ok=True)

    # Save the models
    joblib.dump(baseline_clf, baseline_model_path)
    joblib.dump(advanced_clf, advanced_model_path)
    logging.info("Saved baseline model to %s", baseline_model_path)
    logging.info("Saved advanced model to %s", advanced_model_path)

    return str(baseline_model_path), str(advanced_model_path)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Train sentiment classification models.")
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/features.joblib",
        help="Path to the TF‑IDF features joblib file.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="data/processed/labels.joblib",
        help="Path to the labels joblib file.",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="models/baseline_model.joblib",
        help="Path to save the baseline model.",
    )
    parser.add_argument(
        "--advanced_model",
        type=str,
        default="models/advanced_model.joblib",
        help="Path to save the advanced model.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to use as the test set.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()
    train_models(
        args.features,
        args.labels,
        args.baseline_model,
        args.advanced_model,
        test_size=args.test_size,
        random_state=args.random_state,
    )
