"""
download_data.py
=================

This module handles downloading the customer feedback dataset from an external
source.  By default it points at the Kaggle dataset URL for the
Customer Feedback Dataset.  If you have already downloaded the dataset
manually or placed it in the repository, this script can be skipped.  The
function in this module will download the file and save it to the specified
location, creating any missing directories along the way.

Example
-------
To download the dataset into the data/raw directory run this script as a
module:

>>> python -m download_data --output data/raw/sentiment_analysis.csv

"""

import os
import logging
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# URL for the Kaggle dataset.  Note that Kaggle often requires authentication
# for dataset downloads.  Replace this link with a direct file link if you
# host the dataset elsewhere.
DATASET_URL: str = (
    "https://raw.githubusercontent.com/vishweshsalodkar/customer-feedback-dataset/main/sentiment-analysis.csv"
)



def download_dataset(output_path: str, url: Optional[str] = None) -> str:
    """Download the dataset from the given URL and save it locally.

    Parameters
    ----------
    output_path : str
        Path (including filename) to save the downloaded dataset.
    url : Optional[str], default ``None``
        Override the default dataset URL.  This can be useful if you have
        mirrored the dataset on a different storage service.

    Returns
    -------
    str
        The path to the downloaded file.

    Raises
    ------
    requests.HTTPError
        If the download fails due to a non-200 HTTP status code.
    Exception
        For any other error encountered during the download or file write.
    """
    dataset_url = url or DATASET_URL
    output_path = os.path.expanduser(output_path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Starting dataset download from %s", dataset_url)
    try:
        response = requests.get(dataset_url, timeout=60)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
        logging.info("Dataset downloaded and saved to %s", output_path)
        return output_path
    except Exception as exc:
        logging.error("Failed to download dataset: %s", exc)
        raise


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Download the customer feedback dataset.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/sentiment_analysis.csv",
        help="Output file path where the dataset will be stored.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional custom URL to download the dataset from.",
    )
    args = parser.parse_args()
    download_dataset(args.output, url=args.url)
