"""
feature_engineering.py
======================

This module contains utilities for transforming textual customer feedback into
numerical representations suitable for machine learning.  It provides a
function to create TF‑IDF features, save them to disk, and persist the
fitted vectorizer for later use in training and inference.  Feature
engineering is a critical step that converts raw text into a structured
format which machine learning algorithms can understand.

Functions
---------
create_tfidf_features:
    Fit a TF‑IDF vectorizer on the input text and serialize the resulting
    sparse matrix, labels, and vectorizer object to disk.

"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_tfidf_features(
    input_csv: str,
    features_path: str,
    labels_path: str,
    vectorizer_path: str,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[str, str, str]:
    """Generate TF‑IDF features from a processed dataset and save them to disk.

    Parameters
    ----------
    input_csv : str
        Path to the processed dataset CSV.  Must contain at least two columns:
        'Text' for the review text and 'Sentiment' for the labels.
    features_path : str
        File path to save the sparse TF‑IDF matrix (joblib format).
    labels_path : str
        File path to save the labels array (joblib format).
    vectorizer_path : str
        File path to save the fitted TfidfVectorizer (joblib format).
    max_features : int, default 5000
        The maximum number of features (vocabulary size) to keep.
    ngram_range : Tuple[int, int], default (1, 2)
        The lower and upper boundary of the n‑gram range to use for tokenization.

    Returns
    -------
    Tuple[str, str, str]
        Paths to the saved features, labels, and vectorizer files.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    Exception
        For any other error encountered while reading or writing files.
    """
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Processed dataset {input_csv} not found.")

    logging.info("Loading processed data from %s", input_csv)
    df = pd.read_csv(input_path)
    if "Text" not in df.columns or "Sentiment" not in df.columns:
        raise ValueError("Input CSV must contain 'Text' and 'Sentiment' columns.")

    texts = df["Text"].astype(str).tolist()
    labels = df["Sentiment"].astype(str).tolist()

    logging.info("Fitting TF‑IDF vectorizer (max_features=%d, ngram_range=%s)", max_features, ngram_range)
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words="english")
    features = vectorizer.fit_transform(texts)
    logging.info("TF‑IDF feature matrix shape: %s", features.shape)

    # Ensure output directories exist
    Path(features_path).parent.mkdir(parents=True, exist_ok=True)
    Path(labels_path).parent.mkdir(parents=True, exist_ok=True)
    Path(vectorizer_path).parent.mkdir(parents=True, exist_ok=True)

    # Persist the feature matrix, labels, and vectorizer
    joblib.dump(features, features_path)
    joblib.dump(labels, labels_path)
    joblib.dump(vectorizer, vectorizer_path)
    logging.info("Saved features to %s", features_path)
    logging.info("Saved labels to %s", labels_path)
    logging.info("Saved vectorizer to %s", vectorizer_path)
    return features_path, labels_path, vectorizer_path


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Create TF‑IDF features from processed dataset.")
    parser.add_argument("--input", type=str, default="data/processed/sentiment_analysis_clean.csv", help="Input processed CSV file.")
    parser.add_argument("--features", type=str, default="data/processed/features.joblib", help="Path to save the features joblib file.")
    parser.add_argument("--labels", type=str, default="data/processed/labels.joblib", help="Path to save the labels joblib file.")
    parser.add_argument(
        "--vectorizer",
        type=str,
        default="data/processed/vectorizer.joblib",
        help="Path to save the fitted vectorizer.",
    )
    parser.add_argument("--max_features", type=int, default=5000, help="Maximum number of TF‑IDF features.")
    parser.add_argument(
        "--ngram_min", type=int, default=1, help="Minimum n‑gram length for the vectorizer."
    )
    parser.add_argument(
        "--ngram_max", type=int, default=2, help="Maximum n‑gram length for the vectorizer."
    )
    args = parser.parse_args()
    create_tfidf_features(
        args.input,
        args.features,
        args.labels,
        args.vectorizer,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
    )
