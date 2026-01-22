import os
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(__file__), "o3_cot_reasoning.xlsx")
LABELS_COLUMN = "Label"


def load_labels(path: str = DATA_PATH, column: str = LABELS_COLUMN) -> pd.Series:
    """Load the Labels column from the given Excel file.

    Args:
        path: Absolute or relative path to the Excel file.
        column: Column name containing labels.

    Returns:
        pandas Series of labels.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Excel file not found at: {path}")
    df = pd.read_excel(path)
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found. Available columns: {list(df.columns)}")
    return df[column].dropna()


def get_unique_labels(labels: pd.Series) -> List[str]:
    """Return sorted unique labels from the provided Series."""
    return sorted(set(labels.astype(str)))


def plot_label_distribution(labels: pd.Series, title: str = "Label Distribution") -> Tuple[plt.Figure, plt.Axes]:
    """Plot a bar chart of label counts.

    Args:
        labels: Series of labels.
        title: Plot title.

    Returns:
        (fig, ax) matplotlib objects for further customization or saving.
    """
    counts = labels.astype(str).value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig, ax


def main():
    labels = load_labels()
    unique = get_unique_labels(labels)
    print(f"Unique labels ({len(unique)}): {unique}")
    fig, _ = plot_label_distribution(labels, title="CSSRS Labels Distribution")
    plt.show()


if __name__ == "__main__":
    main()
