import json
import os

import fiftyone as fo
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from sklearn.metrics import balanced_accuracy_score, f1_score

from z_photos.datasets import CanonicalLabel

labels_list = [c.value for c in CanonicalLabel]
label2id = {lbl: i for i, lbl in enumerate(labels_list)}


class LinearHeadModel(nn.Module):
    def __init__(self, input_dim=1152, num_labels=6):
        super().__init__()
        self.num_labels = num_labels
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, inputs_embeds):
        return {"logits": self.classifier(inputs_embeds)}


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
eval_dataset = fo.load_dataset("z_photos-gold-eval-clean")

f1s = []
baccs = []

for fold in range(5):
    # read best checkpoint path
    with open(
        f"checkpoints/ablation_single_split_fold_{fold}/checkpoint-120/trainer_state.json"
    ) as f:
        state = json.load(f)
        best_ckpt = state.get("best_model_checkpoint")

    if not best_ckpt:
        best_ckpt = f"checkpoints/ablation_single_split_fold_{fold}/checkpoint-120"

    model = LinearHeadModel().to(device)
    state_dict = load_file(os.path.join(best_ckpt, "model.safetensors"))
    model.load_state_dict(state_dict)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for sample in eval_dataset.select_fields(
            ["siglip2_embeddings", "canonical_label"]
        ):
            emb = torch.tensor(
                np.array(sample.siglip2_embeddings, dtype=np.float32), device=device
            ).unsqueeze(0)
            logits = model(emb)["logits"].squeeze(0).cpu().numpy()
            y_true.append(label2id[sample.canonical_label])
            y_pred.append(int(np.argmax(logits)))

    f1s.append(f1_score(y_true, y_pred, average="macro"))
    baccs.append(balanced_accuracy_score(y_true, y_pred))

print(f"F1s: {f1s}")
print(f"BAccs: {baccs}")
print(f"Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"Mean BAcc: {np.mean(baccs):.4f} ± {np.std(baccs):.4f}")
