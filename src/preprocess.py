"""
preprocess.py
==============

This module provides functionality to preprocess the raw customer feedback
dataset.  It reads the raw CSV file, removes duplicate entries, handles
missing values, and saves a cleaned version of the dataset into a
processed data directory.  Preprocessing ensures that downstream modeling
steps operate on high-quality, non-redundant data.

Functions
---------
preprocess_dataset:
    Main entry point for preprocessing a dataset from a raw CSV to a
    processed CSV.

"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def preprocess_dataset(input_csv: str, output_csv: str) -> str:
    """Preprocess the raw customer feedback dataset.

    The preprocessing steps include:

    * Dropping duplicate rows.
    * Removing rows with missing text or sentiment labels.
    * Stripping whitespace from text fields.

    Parameters
    ----------
    input_csv : str
        Path to the raw dataset CSV file.
    output_csv : str
        Path to save the processed dataset.

    Returns
    -------
    str
        Path to the processed dataset file.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    Exception
        For any other error encountered while reading or writing the files.
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_csv} does not exist.")

    logging.info("Reading raw dataset from %s", input_csv)
    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        logging.error("Failed to read input CSV: %s", exc)
        raise

    # Basic data sanity checks
    logging.info("Initial dataset shape: %s", df.shape)
    # Remove duplicates
    df = df.drop_duplicates()
    logging.info("Dataset shape after removing duplicates: %s", df.shape)
    # Drop rows with missing values in essential columns
    essential_columns = ["Text", "Sentiment"]
    df = df.dropna(subset=essential_columns)
    logging.info("Dataset shape after dropping missing values: %s", df.shape)
    # Strip whitespace from text
    df["Text"] = df["Text"].astype(str).str.strip()
    # Save processed dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("Processed dataset saved to %s", output_csv)
    return str(output_path)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess the customer feedback dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/sentiment_analysis.csv",
        help="Path to the raw dataset CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/sentiment_analysis_clean.csv",
        help="Path to save the processed dataset.",
    )
    args = parser.parse_args()
    preprocess_dataset(args.input, args.output)
