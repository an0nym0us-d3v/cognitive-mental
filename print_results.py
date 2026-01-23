from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

DATA_FOLDERS = [
    "CSSRS",
    "DepSeverity",
    "Dreaddit",
    "RedSam",
    "SDCNL",
]

TARGET_FILES = [
    "o3_cot_reasoning.xlsx",
    "o3_tot_reasoning.xlsx",
    "o3_few_shot_reasoning.xlsx",
]

REQUIRED_COLUMNS = ["Label", "GPT Label"]


def _to_camel_case(s: str) -> str:
    """Convert a label to uniform CamelCase (letters/digits only)."""
    if s is None:
        return ""
    # Normalize spaces and split on non-alphanumeric boundaries
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", str(s).strip()) if p]
    # Title-case each part and join
    return "".join(p.capitalize() for p in parts)


def print_confusion_matrix_for_file(file_path: Path) -> None:
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to read {file_path}: {e}")
        return

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        print(f"[WARN] {file_path} missing columns: {missing}")
        return

    # Drop rows with missing labels
    sub = df[REQUIRED_COLUMNS].dropna()

    # Ensure labels are comparable (strings) and trimmed
    y_true = sub["Label"].astype(str).str.strip().map(_to_camel_case)
    y_pred = sub["GPT Label"].astype(str).str.strip().map(_to_camel_case)

    # Remove empty after normalization
    valid_mask = (y_true != "") & (y_pred != "")
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        print(f"[WARN] {file_path} has no valid rows after cleaning.")
        return

    # Use pandas crosstab to compute confusion matrix
    labels = sorted(set(y_true.unique()).union(set(y_pred.unique())))
    cm_df = pd.crosstab(y_true, y_pred, rownames=["true"], colnames=["pred"], dropna=False)
    cm_df = cm_df.reindex(index=labels, columns=labels, fill_value=0)

    # Plot and save confusion matrix heatmap
    plt.figure(figsize=(max(6, len(labels)), max(4, len(labels))))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix\n{file_path.name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    out_path = file_path.with_suffix("")
    out_file = out_path.parent / f"{out_path.name}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"[INFO] Saved confusion matrix: {out_file}")


# --- New accuracy computation helpers ---

def compute_accuracy(file_path: Path) -> float:
    """Compute accuracy for a single Excel file using normalized labels.
    Returns NaN if the file cannot be read or has no valid rows.
    """
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to read {file_path}: {e}")
        return float('nan')

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        print(f"[WARN] {file_path} missing columns: {missing}")
        return float('nan')

    sub = df[REQUIRED_COLUMNS].dropna()
    if sub.empty:
        return float('nan')

    y_true = sub["Label"].astype(str).str.strip().map(_to_camel_case)
    y_pred = sub["GPT Label"].astype(str).str.strip().map(_to_camel_case)

    valid_mask = (y_true != "") & (y_pred != "")
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return float('nan')

    acc = (y_true == y_pred).mean()
    return float(acc)


def aggregate_accuracies(root: Path) -> pd.DataFrame:
    """Aggregate accuracies across datasets and target files.
    Returns a DataFrame with columns: Dataset, File, Accuracy.
    """
    records = []
    for dataset in DATA_FOLDERS:
        dpath = root / dataset
        if not dpath.exists() or not dpath.is_dir():
            print(f"[INFO] Skipping non-folder or missing path: {dpath}")
            continue
        for fname in TARGET_FILES:
            fpath = dpath / fname
            if not (fpath.exists() and fpath.is_file()):
                print(f"[INFO] Not found: {fpath}")
                continue
            acc = compute_accuracy(fpath)
            records.append({
                "Dataset": dataset,
                "File": fname,
                "Accuracy": acc,
            })
    return pd.DataFrame.from_records(records)


def plot_accuracies(df: pd.DataFrame, out_path: Path) -> None:
    """Plot multi-line chart of accuracy per dataset for each file."""
    if df.empty:
        print("[WARN] No accuracy data to plot.")
        return
    # Sort datasets in the given order
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=DATA_FOLDERS, ordered=True)
    df = df.sort_values(["File", "Dataset"])  # ensure consistent line ordering

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Dataset", y="Accuracy", hue="File", marker="o")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Dataset and File")
    plt.legend(title="Excel File")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved accuracy plot: {out_path}")


def main():
    root = Path(__file__).parent

    # Compute and plot accuracies
    acc_df = aggregate_accuracies(root)
    plot_accuracies(acc_df, root / "accuracy_plot.png")

    for folder in DATA_FOLDERS:
        folder_path = root / folder
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"[INFO] Skipping non-folder or missing path: {folder_path}")
            continue

        for fname in TARGET_FILES:
            fpath = folder_path / fname
            if fpath.exists() and fpath.is_file():
                # print_confusion_matrix_for_file(fpath)
                pass
            else:
                print(f"[INFO] Not found: {fpath}")


if __name__ == "__main__":
    main()
