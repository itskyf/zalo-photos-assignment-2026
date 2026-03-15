import os

import fiftyone as fo
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    # Load dataset
    dataset_name = "z_photos-silver-train"
    if dataset_name not in fo.list_datasets():
        print(f"Dataset {dataset_name} not found.")
        return

    dataset = fo.load_dataset(dataset_name)

    # Extract clean_score and canonical_label
    # Lấy tất cả các mẫu có clean_score, nếu không có label thì để là "unlabeled"
    view = dataset.exists("clean_score")

    data = []
    for sample in view:
        label = (
            sample.canonical_label
            if sample.canonical_label is not None
            else "unlabeled"
        )
        data.append({"clean_score": sample.clean_score, "class": label})

    if not data:
        print("No data found for visualization.")
        return

    df = pd.DataFrame(data)

    # Sắp xếp theo tên class để plot đẹp hơn
    df = df.sort_values("class")

    # Visualization
    plt.figure(figsize=(12, 6))

    # Chỉ vẽ Boxplot
    sns.boxplot(x="class", y="clean_score", data=df, palette="Set3")
    # Thêm các điểm dữ liệu (stripplot) để thấy rõ mật độ
    sns.stripplot(
        x="class", y="clean_score", data=df, color="orange", alpha=0.3, jitter=True
    )

    plt.title("Boxplot of Clean Score by Class (including unlabeled)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save the plot
    output_path = "images/clean_score_boxplot.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()
