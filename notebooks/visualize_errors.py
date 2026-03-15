import math
from pathlib import Path

import fiftyone as fo
import matplotlib.pyplot as plt
from fiftyone import ViewField as F


def main():
    dataset_name = "z_photos-bronze"
    if dataset_name not in fo.list_datasets():
        print(f"Error: Dataset '{dataset_name}' not found.")
        return

    ds = fo.load_dataset(dataset_name)

    # 1. Create a FiftyOne view for High-Confidence Errors
    # Condition: Prediction does not match Ground Truth
    error_view = ds.match(F("ground_truth.label") != F("siglip2_pred.label"))

    # Sort by prediction confidence descending
    error_view = error_view.sort_by("siglip2_pred.confidence", reverse=True)

    # Save the view so it can be selected in the FiftyOne App
    view_name = "high_confidence_errors"
    ds.save_view(view_name, error_view)

    print(f"Saved view '{view_name}' to dataset '{dataset_name}'.")
    print(
        "In the FiftyOne App, click the 'Unsaved view' dropdown (top left) and select 'high_confidence_errors'."
    )

    # 2. Plot the top high-confidence errors
    top_k = 8
    top_errors = error_view.limit(top_k)

    n_samples = len(top_errors)
    if n_samples == 0:
        print("No errors found!")
        return

    cols = 4
    rows = math.ceil(n_samples / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if rows == 1:
        axes = [axes]

    axes_flat = [
        ax
        for row in axes
        for ax in (row if isinstance(row, (list, tuple, type(axes))) else [row])
    ]

    for i, sample in enumerate(top_errors):
        ax = axes_flat[i]

        # Load image
        img = plt.imread(sample.filepath)
        ax.imshow(img)

        # Format title
        gt_label = sample.ground_truth.label
        pred_label = sample.siglip2_pred.label
        conf = sample.siglip2_pred.confidence

        title = f"GT: {gt_label}\nPred: {pred_label} ({conf:.2f})"
        ax.set_title(title, fontsize=10, color="red")
        ax.axis("off")

    # Turn off unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()

    # Save the plot
    output_dir = Path("images")
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "top_high_confidence_errors.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot of top {n_samples} high-confidence errors to: {out_path}")


if __name__ == "__main__":
    main()
