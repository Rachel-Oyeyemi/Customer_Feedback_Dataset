"""
utils.py
========

This module contains utility functions used throughout the project.  These
functions include convenience wrappers for loading data, plotting confusion
matrices, and saving figures to disk.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    logging.info("Loading CSV from %s", path)
    return pd.read_csv(path)


def save_figure(fig, output_path: str) -> str:
    """Save a matplotlib figure to disk.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    output_path : str
        Destination path for the saved image.

    Returns
    -------
    str
        The path to the saved image file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    logging.info("Saved figure to %s", output_path)
    return output_path


def plot_confusion_matrix(y_true: List[str], y_pred: List[str], classes: List[str], title: str = "Confusion Matrix"):
    """Create a confusion matrix plot for classification results.

    Parameters
    ----------
    y_true : list of str
        True labels.
    y_pred : list of str
        Predicted labels.
    classes : list of str
        List of class names in the order they should appear on the axes.
    title : str, optional
        Title for the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The confusion matrix figure.
    """
    logging.info("Generating confusion matrix plot")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=classes, ax=ax, cmap="Blues")
    ax.set_title(title)
    return fig
