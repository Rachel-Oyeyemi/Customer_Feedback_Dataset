"""
predict.py
==========

This module defines helper functions for loading a trained sentiment
classification model and vectoriser and using them to predict the sentiment
of new text inputs.  It can be used programmatically or via the command line.

Functions
---------
predict_sentiment:
    Given a text string, return the predicted sentiment label.
"""

import logging
from pathlib import Path
from typing import List

import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_model(model_path: str):
    """Load a trained model from disk.

    Parameters
    ----------
    model_path : str
        Path to the saved model joblib file.

    Returns
    -------
    object
        The loaded model.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file {model_path} not found.")
    return joblib.load(model_path)


def load_vectorizer(vectorizer_path: str):
    """Load a fitted TfidfVectorizer from disk.

    Parameters
    ----------
    vectorizer_path : str
        Path to the saved vectoriser joblib file.

    Returns
    -------
    TfidfVectorizer
        The loaded vectoriser.
    """
    if not Path(vectorizer_path).exists():
        raise FileNotFoundError(f"Vectoriser file {vectorizer_path} not found.")
    return joblib.load(vectorizer_path)


def predict_sentiment(
    texts: List[str], model, vectorizer
) -> List[str]:
    """Predict sentiment labels for a list of input texts.

    Parameters
    ----------
    texts : List[str]
        List of raw text strings to classify.
    model : object
        Trained classification model.
    vectorizer : TfidfVectorizer
        Fitted TF‑IDF vectoriser used during model training.

    Returns
    -------
    List[str]
        List of predicted sentiment labels.
    """
    logging.info("Transforming input texts using TF‑IDF vectoriser")
    features = vectorizer.transform(texts)
    logging.info("Predicting sentiment labels")
    return model.predict(features).tolist()


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Predict sentiment of input text using a trained model.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/baseline_model.joblib",
        help="Path to the trained model.",
    )
    parser.add_argument(
        "--vectorizer",
        type=str,
        default="data/processed/vectorizer.joblib",
        help="Path to the TF‑IDF vectoriser.",
    )
    parser.add_argument(
        "text",
        type=str,
        nargs="+",
        help="One or more text strings to classify (enclose in quotes).",
    )
    args = parser.parse_args()

    model = load_model(args.model)
    vectoriser = load_vectorizer(args.vectorizer)
    predictions = predict_sentiment(args.text, model, vectoriser)
    for t, p in zip(args.text, predictions):
        print(f"Input: {t}\nPredicted Sentiment: {p}\n")
