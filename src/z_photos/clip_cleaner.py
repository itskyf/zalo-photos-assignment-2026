import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, TensorType
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import PaddingStrategy

from z_photos.datasets import LABEL_REGISTRY, SilverField


class SigLIP2Cleaner:
    """CLIPCleaner-style data cleaning using SigLIP2 and OOF Logistic
    Regression."""

    def __init__(
        self,
        model_name: str = "google/siglip2-so400m-patch16-384",
        device_map: str = "auto",
    ):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map=device_map).eval()

    @torch.inference_mode()
    def get_image_embeddings(
        self, image_paths: list[str], batch_size: int = 16
    ) -> np.ndarray:
        """Extract image embeddings using SigLIP2."""
        all_embeddings = []
        for i in tqdm(
            range(0, len(image_paths), batch_size), desc="Extracting embeddings"
        ):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]

            image_inputs = self.processor(
                images=images, return_tensors=TensorType.PYTORCH
            )

            # Move to device and cast to model dtype
            image_inputs = {
                k: v.to(
                    self.model.device,
                    dtype=self.model.dtype if v.is_floating_point() else None,
                )
                for k, v in image_inputs.items()
            }

            outputs = self.model.get_image_features(**image_inputs)

            if isinstance(outputs, BaseModelOutputWithPooling):
                if outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output
                else:
                    raise ValueError("Image features are not available")
            else:
                embeddings = outputs

            # SigLIP2 embeddings are designed to work unnormalized,
            # but for LR and some scoring we might still want them on CPU
            all_embeddings.append(embeddings.detach().cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    @torch.inference_mode()
    def compute_zs_scores(
        self, embeddings: np.ndarray, labels: list[str]
    ) -> np.ndarray:
        """Compute zero-shot scores for ground-truth classes using a bilingual
        prompt bank."""
        # Filter for labels that are in our registry
        class_to_idx = {
            m.canonical_label.value: i for i, m in enumerate(LABEL_REGISTRY)
        }

        # Prepare prompts
        prompts_en = [f"a photo of a {m.en_name}" for m in LABEL_REGISTRY]
        prompts_vi = [f"một bức ảnh về {m.vi_name}" for m in LABEL_REGISTRY]

        # Process text inputs
        text_inputs_en = self.processor(
            text=prompts_en,
            padding=PaddingStrategy.MAX_LENGTH,
            return_tensors=TensorType.PYTORCH,
        ).to(self.model.device)

        text_inputs_vi = self.processor(
            text=prompts_vi,
            padding=PaddingStrategy.MAX_LENGTH,
            return_tensors=TensorType.PYTORCH,
        ).to(self.model.device)

        out_en = self.model.get_text_features(**text_inputs_en)
        out_vi = self.model.get_text_features(**text_inputs_vi)

        def _extract(out):
            if isinstance(out, BaseModelOutputWithPooling):
                return out.pooler_output
            return out

        text_emb_en = _extract(out_en)
        text_emb_vi = _extract(out_vi)

        if text_emb_en is None or text_emb_vi is None:
            raise ValueError("Text features are not available")

        # SigLIP2 prefers unnormalized inputs but averaged similarities
        # Following zoo.py: no normalization per-se, but we average EN and VI
        text_features = (text_emb_en + text_emb_vi) / 2.0

        # Compute similarity (N, K)
        img_features = torch.from_numpy(embeddings).to(
            self.model.device, dtype=torch.float32
        )
        text_features = text_features.to(dtype=torch.float32)

        # SigLIP uses a learned temperature and bias
        logit_scale = getattr(self.model, "logit_scale", torch.tensor(10.0)).exp()
        logits = (img_features @ text_features.t()) * logit_scale

        # Softmax for normalized class probabilities
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Get probability of the ground-truth class
        # If label is None or not in registry, we return 0.0 or handle accordingly
        zs_scores = []
        for i, label in enumerate(labels):
            if label in class_to_idx:
                zs_scores.append(probs[i, class_to_idx[label]])
            else:
                zs_scores.append(0.0)

        return np.array(zs_scores)

    def compute_lr_oof_scores(
        self, embeddings: np.ndarray, labels: list[str], n_splits: int = 5
    ) -> np.ndarray:
        """Compute Out-of-Fold Logistic Regression probabilities."""
        labels_arr = np.array(labels)
        oof_probs = np.zeros(len(labels))

        # Only train on samples with valid labels
        valid_mask = np.array([label is not None for label in labels])
        if not any(valid_mask):
            return oof_probs

        valid_embeddings = embeddings[valid_mask]
        valid_labels = labels_arr[valid_mask]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Initialize OOF for valid samples
        valid_oof_probs = np.zeros(len(valid_labels))

        for train_idx, val_idx in skf.split(valid_embeddings, valid_labels):
            x_train, x_val = valid_embeddings[train_idx], valid_embeddings[val_idx]
            y_train = valid_labels[train_idx]

            lr = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )
            lr.fit(x_train, y_train)

            probs = lr.predict_proba(x_val)

            class_to_idx = {cls: i for i, cls in enumerate(lr.classes_)}
            val_labels_fold = valid_labels[val_idx]
            val_label_indices = [class_to_idx[label] for label in val_labels_fold]

            valid_oof_probs[val_idx] = probs[np.arange(len(val_idx)), val_label_indices]

        # Map back to full array
        oof_probs[valid_mask] = valid_oof_probs
        return oof_probs

    def combine_scores(
        self, zs_scores: np.ndarray, lr_scores: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """Combine Zero-shot and LR scores."""
        return alpha * zs_scores + (1 - alpha) * lr_scores


def apply_clip_cleaner(dataset, cleaner: SigLIP2Cleaner = None, alpha: float = 0.4):
    """Apply cleaning pipeline to a FiftyOne dataset (i.e. the silver train
    clone)."""
    if cleaner is None:
        cleaner = SigLIP2Cleaner()

    image_paths = dataset.values("filepath")
    labels = dataset.values("ground_truth.label")

    embeddings = cleaner.get_image_embeddings(image_paths)
    dataset.set_values(
        SilverField.SIGLIP2_EMB, [list(e.astype(float)) for e in embeddings]
    )

    print("Computing Zero-shot scores...")
    zs_scores = cleaner.compute_zs_scores(embeddings, labels)
    dataset.set_values(SilverField.ZS_SCORE, [float(s) for s in zs_scores])

    print("Computing LR OOF scores...")
    lr_scores = cleaner.compute_lr_oof_scores(embeddings, labels)
    dataset.set_values(SilverField.LR_OOF_SCORE, [float(s) for s in lr_scores])

    print("Combining scores...")
    clean_scores = cleaner.combine_scores(zs_scores, lr_scores, alpha=alpha)
    dataset.set_values(SilverField.CLEAN_SCORE, [float(s) for s in clean_scores])

    # Preliminary weight
    dataset.set_values(
        SilverField.SAMPLE_WEIGHT_PRELIM, [float(s) for s in clean_scores]
    )

    dataset.save()
    print(f"Cleaning scores applied to {len(dataset)} samples.")
