import fiftyone as fo


def main():
    # Use gold-train if it exists, as it has the final sample weights
    if "z_photos-gold-train" in fo.list_datasets():
        dataset_name = "z_photos-gold-train"
    elif "z_photos-silver-train" in fo.list_datasets():
        dataset_name = "z_photos-silver-train"
    else:
        print(
            "Required datasets not found. Please run notebooks/experiment-02-silver.ipynb first."
        )
        return

    dataset = fo.load_dataset(dataset_name)

    # Filter for samples that have a ground_truth label to see ambiguous/noisy but labeled examples
    # Sort by clean_score ascending (most noisy first)
    view = dataset.exists("ground_truth").sort_by("clean_score", reverse=False)

    print(f"Dataset: {dataset_name}")
    print(f"Total samples with labels: {len(view)}")
    print("Launching FiftyOne App...")
    print("---")
    print("GUIDE TO VISUALIZE AMBIGUOUS IMAGES:")
    print("1. The app will open with samples sorted by 'clean_score' (lowest first).")
    print("2. Select the first few images (e.g., indices 0, 1, 2).")
    print("3. In the sidebar, look at the following fields:")
    print("   - 'ground_truth.label': The assigned label.")
    print("   - 'zs_score': Probability from zero-shot SigLIP2 (prior knowledge).")
    print(
        "   - 'lr_oof_score': Probability from Out-of-Fold Logistic Regression (data-driven)."
    )
    print("   - 'clean_score': Combined score used for down-weighting.")
    print("4. Notice how images with low 'clean_score' often have conflicting content")
    print("   (e.g., an 'other' image that looks like 'gathering', or vice versa).")
    print("---")

    session = fo.launch_app(view)
    session.wait()


if __name__ == "__main__":
    main()
