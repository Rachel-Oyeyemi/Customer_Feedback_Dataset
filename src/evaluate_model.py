"""
evaluate_model.py
=================

This module provides functionality to evaluate trained sentiment classification
models on a set of TF‑IDF features and labels.  It computes classification
metrics such as accuracy, precision, recall, F1‑score, and a confusion matrix.
Results can be returned as a dictionary for programmatic use or saved to a
JSON file for reporting.

Functions
---------
evaluate_model:
    Evaluate a trained model and return performance metrics.
save_evaluation_report:
    Save evaluation metrics and confusion matrix to a JSON file.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def evaluate_model(model_path: str, features_path: str, labels_path: str) -> Dict[str, Any]:
    """Evaluate a trained model on the given features and labels.

    Parameters
    ----------
    model_path : str
        Path to the trained model file (joblib).
    features_path : str
        Path to the TF‑IDF features joblib file.
    labels_path : str
        Path to the labels joblib file.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing accuracy, precision, recall, F1‑score, a
        classification report (string), and the confusion matrix.

    Raises
    ------
    FileNotFoundError
        If any of the input files cannot be found.
    Exception
        For any other error during evaluation.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file {model_path} not found.")
    if not Path(features_path).exists():
        raise FileNotFoundError(f"Features file {features_path} not found.")
    if not Path(labels_path).exists():
        raise FileNotFoundError(f"Labels file {labels_path} not found.")

    logging.info("Loading model and data for evaluation")
    model = joblib.load(model_path)
    X = joblib.load(features_path)
    y_true = joblib.load(labels_path)

    logging.info("Generating predictions")
    y_pred = model.predict(X)

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    class_report = classification_report(y_true, y_pred, output_dict=False)
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix,
    }

    logging.info(
        "Model evaluation completed – Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
        accuracy,
        precision,
        recall,
        f1,
    )

    return metrics


def save_evaluation_report(metrics: Dict[str, Any], output_path: str) -> str:
    """Save evaluation metrics to a JSON file.

    Parameters
    ----------
    metrics : dict
        Dictionary of evaluation metrics returned by `evaluate_model`.
    output_path : str
        Path to save the JSON report.

    Returns
    -------
    str
        Path to the saved report.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=4)
    logging.info("Evaluation report saved to %s", output_path)
    return output_path


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained sentiment classification model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model joblib file.",
    )
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
        "--output",
        type=str,
        default="models/evaluation_report.json",
        help="Path to save the evaluation report JSON.",
    )
    args = parser.parse_args()
    metrics = evaluate_model(args.model, args.features, args.labels)
    save_evaluation_report(metrics, args.output)
