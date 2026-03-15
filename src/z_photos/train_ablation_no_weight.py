from datetime import datetime

import dagshub
import fiftyone as fo
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from z_photos.datasets import CanonicalLabel

# ==========================================
# 1. SETUP DAGSHUB & MLFLOW
# ==========================================
dagshub.init(repo_owner="cykanp", repo_name="zalo-photos-assignment-2026", mlflow=True)
mlflow.set_experiment("Stage4_Ablation_No_Weighting")

# ==========================================
# 2. DEFINITIONS (Model, Dataset, Trainer)
# ==========================================
labels_list = [c.value for c in CanonicalLabel]
label2id = {lbl: i for i, lbl in enumerate(labels_list)}
id2label = dict(enumerate(labels_list))


class FeatureDataset(Dataset):
    def __init__(self, fo_view, label_to_id: dict[str, int]):
        self.samples = []
        # Ablation: Without CLIPCleaner weighting means sample_weight is always 1.0
        for s in fo_view.select_fields(["siglip2_emb", "canonical_label", "id"]):
            emb = np.array(s.siglip2_emb, dtype=np.float32)
            label_id = label_to_id[s.canonical_label]
            weight = 1.0  # Force weight to 1.0

            self.samples.append(
                {
                    "inputs_embeds": emb,
                    "labels": label_id,
                    "sample_weight": weight,
                    "sample_id": str(s.id),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    return {
        "inputs_embeds": torch.tensor(
            np.stack([x["inputs_embeds"] for x in batch]), dtype=torch.float32
        ),
        "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
        "sample_weight": torch.tensor(
            [x["sample_weight"] for x in batch], dtype=torch.float32
        ),
        "sample_id": [x["sample_id"] for x in batch],
    }


class LinearHeadModel(nn.Module):
    def __init__(self, input_dim=1152, num_labels=6):
        super().__init__()
        self.num_labels = num_labels
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, inputs_embeds, labels=None, sample_weight=None, sample_id=None):
        logits = self.classifier(inputs_embeds)
        return {"logits": logits}


class CustomFeatureTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        sample_weights = inputs.pop("sample_weight", None)
        inputs.pop("sample_id", None)

        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            class_weights = self.class_weights.to(logits.device)
        else:
            class_weights = None

        loss_fct = nn.CrossEntropyLoss(
            weight=class_weights, reduction="none", label_smoothing=0.1
        )
        loss = loss_fct(logits, labels)

        if sample_weights is not None:
            loss = loss * sample_weights

        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "balanced_acc": balanced_accuracy_score(labels, preds),
        "accuracy": accuracy_score(labels, preds),
    }


# ==========================================
# 3. EVALUATION FUNCTION
# ==========================================
def evaluate_on_clean_set(model, device):
    print("\n=== Evaluating on z_photos-gold-eval-clean ===")
    eval_dataset = fo.load_dataset("z_photos-gold-eval-clean")

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for sample in eval_dataset.select_fields(
            ["siglip2_embeddings", "canonical_label"]
        ):
            # Use siglip2_embeddings (VectorField) from eval set
            emb = torch.tensor(
                np.array(sample.siglip2_embeddings, dtype=np.float32), device=device
            ).unsqueeze(0)
            outputs = model(emb)
            logits = outputs["logits"].squeeze(0).cpu().numpy()
            pred_id = int(np.argmax(logits))

            y_true.append(label2id[sample.canonical_label])
            y_pred.append(pred_id)

    m_f1 = f1_score(y_true, y_pred, average="macro")
    b_acc = balanced_accuracy_score(y_true, y_pred)

    print("Final Evaluation Results:")
    print(f"Macro-F1: {m_f1:.4f}")
    print(f"Balanced Acc: {b_acc:.4f}")

    mlflow.log_metric("final_eval_macro_f1", m_f1)
    mlflow.log_metric("final_eval_balanced_acc", b_acc)

    return m_f1, b_acc


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    dataset_name = "z_photos-gold-train"
    if dataset_name not in fo.list_datasets():
        raise ValueError(
            f"Dataset {dataset_name} NOT found! Please run main.ipynb first."
        )

    dataset = fo.load_dataset(dataset_name)
    print(f"Loaded {len(dataset)} samples from {dataset_name}")

    # ==========================================
    # 5. CROSS VALIDATION
    # ==========================================
    from sklearn.utils.class_weight import compute_class_weight

    view = dataset.match(
        (fo.ViewField("keep_for_train")) & (fo.ViewField("canonical_label").exists())
    )
    print(
        f"Training on {len(view)} samples (keep_for_train=True, valid canonical_label)"
    )

    labels = view.values("canonical_label")

    # Generate Folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    view.set_values("fold_id", [None] * len(view))
    fold_ids = [None] * len(view)
    for fold_idx, (_train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels)
    ):
        for vi in val_idx:
            fold_ids[vi] = fold_idx
    view.set_values("fold_id", fold_ids)

    all_f1 = []
    all_bal_acc = []

    run_ts = datetime.now().strftime("%m%d-%H%M")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    for fold in range(5):
        print(f"\n=== Training Fold {fold} ===")

        train_view = view.match(fo.ViewField("fold_id") != fold)
        val_view = view.match(fo.ViewField("fold_id") == fold)

        train_ds = FeatureDataset(train_view, label2id)
        val_ds = FeatureDataset(val_view, label2id)

        y_train = [train_ds[i]["labels"] for i in range(len(train_ds))]

        c_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        weight_map = dict(zip(np.unique(y_train), c_weights, strict=True))
        final_class_weights = [weight_map.get(i, 1.0) for i in range(len(labels_list))]

        model = LinearHeadModel(input_dim=1152, num_labels=len(labels_list)).to(device)

        args = TrainingArguments(
            output_dir=f"./checkpoints/ablation_no_weight_cv_fold_{fold}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=1e-3,
            auto_find_batch_size=True,
            num_train_epochs=5,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            report_to="mlflow",
            run_name=f"ablation-no-weight-fold-{fold}-{run_ts}",
        )

        trainer = CustomFeatureTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            class_weights=final_class_weights,
        )

        trainer.train()

        val_outputs = trainer.predict(val_ds)
        all_f1.append(val_outputs.metrics["test_macro_f1"])
        all_bal_acc.append(val_outputs.metrics["test_balanced_acc"])

        print(f"Fold {fold} Macro F1: {val_outputs.metrics['test_macro_f1']:.4f}")

    print("\n=== CV Completed ===")
    print(f"Mean Macro F1: {np.mean(all_f1):.4f}")
    print(f"Mean Balanced Acc: {np.mean(all_bal_acc):.4f}")

    # Optional: Train a final model on all data
    print("\n=== Training Final Model ===")
    full_ds = FeatureDataset(view, label2id)
    y_full = [full_ds[i]["labels"] for i in range(len(full_ds))]

    c_weights = compute_class_weight("balanced", classes=np.unique(y_full), y=y_full)
    weight_map = dict(zip(np.unique(y_full), c_weights, strict=True))
    final_class_weights = [weight_map.get(i, 1.0) for i in range(len(labels_list))]

    model_final = LinearHeadModel(input_dim=1152, num_labels=len(labels_list)).to(
        device
    )
    args_final = TrainingArguments(
        output_dir="./checkpoints/ablation_no_weight_final",
        eval_strategy="no",
        save_strategy="no",
        learning_rate=1e-3,
        auto_find_batch_size=True,
        num_train_epochs=5,
        report_to="mlflow",
        run_name=f"ablation-no-weight-final-{run_ts}",
    )

    with mlflow.start_run(run_name=f"ablation-no-weight-final-eval-{run_ts}"):
        trainer_final = CustomFeatureTrainer(
            model=model_final,
            args=args_final,
            train_dataset=full_ds,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            class_weights=final_class_weights,
        )

        trainer_final.train()
        trainer_final.save_model("./checkpoints/ablation_no_weight_final")

        # Evaluate
        evaluate_on_clean_set(model_final, device)

    print("Done! Checkpoints saved to ./checkpoints/ablation_no_weight_final")


if __name__ == "__main__":
    main()
