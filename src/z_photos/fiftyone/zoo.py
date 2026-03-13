# https://github.com/harpreetsahota204/siglip2/blob/main/zoo.py

import logging

import fiftyone.core.models as fom
import fiftyone.utils.torch as fout
import numpy as np
import torch
from transformers import AutoProcessor, SiglipModel, TensorType
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import PaddingStrategy
from transformers.utils.import_utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)


class SigLIP2Config(fout.TorchImageModelConfig):
    """This config class extends TorchImageModelConfig to provide specific
    parameters needed for text-image similarity search.

    Args:
        model_path (str): Path to the model's weights on disk or HuggingFace model ID.

        text_prompt (str): Optional baseline text prompt to use for classification.
            Defaults to "".
    """

    def __init__(self, d):
        """Initialize the configuration.

        Args:
            d: A dictionary containing the configuration parameters
        """
        # Processor handles preprocessing, so use raw inputs
        if "raw_inputs" not in d:
            d["raw_inputs"] = True

        # Only set up output processor if classes provided (for classification)
        if (
            "classes" in d
            and d["classes"] is not None
            and len(d["classes"]) > 0
            and "output_processor_cls" not in d
        ):
            d["output_processor_cls"] = "z_photos.fiftyone.zoo.SigLIPOutputProcessor"

        super().__init__(d)

        # Path to model weights or HuggingFace model ID
        self.model_path = self.parse_string(d, "model_path", default="")
        # Optional base text prompt
        self.text_prompt = self.parse_string(d, "text_prompt", default="")

        self.device_map = d.get("device_map", "auto")
        self.dtype = d.get("dtype", None)


class SigLIPOutputProcessor:
    """Custom output processor for SigLIP to handle numerical stability.

    FiftyOne's default ClassifierOutputProcessor uses naive np.exp()
    which overflows with SigLIP's unnormalized and highly scaled logits.
    """

    def __init__(self, classes):
        self.classes = classes
        self.store_logits = False

    def __call__(self, logits, frame_size=None, confidence_thresh=None):
        import fiftyone.core.labels as fol

        # PyTorch softmax is numerically stable
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()

        preds = []
        predictions = np.argmax(probs, axis=1)
        scores = np.max(probs, axis=1)

        for prediction, score, logit in zip(
            predictions, scores, logits_np, strict=False
        ):
            label = self.classes[prediction]
            if confidence_thresh is not None and score < confidence_thresh:
                classification = fol.Classification()
            else:
                classification = fol.Classification(
                    label=label,
                    confidence=float(score),
                )
                if self.store_logits:
                    classification.logits = logit
            preds.append(classification)
        return preds


class SigLIP2(fout.TorchImageModel, fom.PromptMixin):
    """This model leverages a vision-language model to create embeddings for
    both images and text in a shared vector space, enabling text-image
    similarity search.

    The model can:
    1. Embed images into a vector space
    2. Embed text queries into the same vector space
    3. Calculate similarity between images and text
    4. Zero-shot classification: comparing image embeddings to class name embeddings

    It extends TorchImageModel for image processing capabilities and PromptMixin to
    enable text embedding capabilities.
    """

    def __init__(self, config):
        """Initialize the model.

        Args:
            config: A Config instance containing model parameters
        """
        if isinstance(config, dict):
            config = SigLIP2Config(config)

        # Initialize parent classes
        super().__init__(config)

        # Store config parameters as instance variables for easier access
        self._text_features = None  # Cached text features for classification

        # Storage for the last computed embeddings (needed for FiftyOne API)
        self._last_computed_embeddings = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit - clear GPU memory cache."""
        # Clear cache based on device type (don't move model to CPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return False

    @property
    def has_embeddings(self):
        """Whether this instance can generate embeddings.

        Returns:
            bool: Always True for this model as embedding generation is supported
        """
        return True

    @property
    def can_embed_prompts(self):
        """Whether this instance can embed text prompts.

        Returns:
            bool: Always True for this model as text embedding is supported
        """
        return True

    @property
    def classes(self):
        """The list of class labels for the model."""
        return self._classes

    @classes.setter
    def classes(self, value):
        """Set new classes and invalidate cached text features."""
        self._classes = value
        self._text_features = None  # Invalidate cache

        # Rebuild output processor if classes are provided
        if value is not None and len(value) > 0:
            self._output_processor = SigLIPOutputProcessor(classes=value)
        else:
            self._output_processor = None

    @property
    def text_prompt(self):
        """The text prompt prefix for classification."""
        return self.config.text_prompt

    @text_prompt.setter
    def text_prompt(self, value):
        """Set new text prompt and invalidate cached text features."""
        self.config.text_prompt = value
        self._text_features = None  # Invalidate cache

    def _load_model(self, config):
        """Load the model and processor from disk or HuggingFace.

        This method initializes both the processor (for tokenization and image
        preprocessing) and the model itself, configuring them for inference.

        Args:
            config: Config instance containing model parameters

        Returns:
            The loaded PyTorch model ready for inference
        """

        # Load the model and processor from HuggingFace
        model_path = config.model_path

        model_kwargs = {
            "device_map": config.device_map,
        }
        if config.dtype is not None:
            model_kwargs["dtype"] = config.dtype

        # Set optimizations based on device capabilities
        if (
            self.device == "cuda"
            and torch.cuda.is_available()
            and is_flash_attn_2_available()
        ):
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Initialize processor and model
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

        self.model = SiglipModel.from_pretrained(model_path, **model_kwargs)
        self.model.eval()
        return self.model

    def _get_text_features(self):
        """Get or compute text features for classification.

        Creates embeddings for each class by combining text_prompt with class names.

        Returns:
            numpy array: Text embeddings with shape (num_classes, embedding_dim)
        """
        # Check if text features are already computed and cached
        if self._text_features is None:
            # Create prompts for each class
            prompts = [f"{self.config.text_prompt} {c}" for c in self.classes]
            # Compute and cache the text features
            self._text_features = self._embed_prompts(prompts)

        # Return the cached features
        return self._text_features

    def _embed_prompts(self, prompts):
        """Embed text prompts for similarity search.

        SigLIP2 uses unnormalized embeddings with sigmoid loss, unlike CLIP.

        Args:
            prompts: List of text prompts to embed

        Returns:
            numpy array: Unnormalized embeddings for the prompts
        """
        # Process text inputs using processor (not tokenizer directly)
        # This ensures consistent preprocessing with native usage
        text_inputs = self.processor(
            text=prompts,
            padding=PaddingStrategy.MAX_LENGTH,
            return_tensors=TensorType.PYTORCH,
        ).to(self.model.device)

        # Get text features
        with torch.inference_mode():
            text_features = self.model.get_text_features(**text_inputs)

        if isinstance(text_features, BaseModelOutputWithPooling):
            if text_features.pooler_output is not None:
                text_features = text_features.pooler_output
            else:
                raise ValueError("Image features are not available")

        # NO NORMALIZATION for SigLIP2 - embeddings are designed to work unnormalized
        # Return as CPU numpy array for FiftyOne compatibility
        return text_features.detach().cpu().numpy()

    def embed_prompt(self, prompt):
        """Embed a single text prompt.

        Args:
            prompt: Text prompt to embed

        Returns:
            numpy array: Embedding for the prompt
        """
        # Format prompt with the template (done inside _embed_prompts)
        # Embed the single prompt by calling _embed_prompts with a list
        embeddings = self._embed_prompts([prompt])
        # Return the first (and only) embedding
        return embeddings[0]

    def embed_prompts(self, prompts):
        """Embed multiple text prompts.

        Args:
            prompts: List of text prompts to embed

        Returns:
            numpy array: Embeddings for the prompts
        """
        # Directly call _embed_prompts which handles batch processing
        return self._embed_prompts(prompts)

    def embed_images(self, imgs):
        """Embed a batch of images.

        SigLIP2 uses unnormalized embeddings with sigmoid loss, unlike CLIP.

        Args:
            imgs: List of images to embed (PIL images or PyTorch tensors)

        Returns:
            numpy array: Unnormalized embeddings for the images
        """

        # Process images
        image_inputs = self.processor(images=imgs, return_tensors=TensorType.PYTORCH)

        # Move to device and cast to model dtype in one go
        image_inputs = {
            k: v.to(
                self.model.device,
                dtype=self.model.dtype if v.is_floating_point() else None,
            )
            for k, v in image_inputs.items()
        }

        # Get image features
        with torch.inference_mode():
            image_features = self.model.get_image_features(**image_inputs)

        if isinstance(image_features, BaseModelOutputWithPooling):
            if image_features.pooler_output is not None:
                image_features = image_features.pooler_output
            else:
                raise ValueError("Image features are not available")

        # NO NORMALIZATION for SigLIP2 - embeddings are designed to work unnormalized

        # Cache the embeddings for get_embeddings() method
        self._last_computed_embeddings = image_features

        # Return as CPU numpy array for FiftyOne compatibility
        return image_features.detach().cpu().numpy()

    def embed(self, img):
        """Embed a single image.

        Implementation of TorchEmbeddingsMixin.embed() method.

        Args:
            img: PIL image or PyTorch tensor to embed

        Returns:
            numpy array: Embedding for the image
        """
        # Convert single image to a list for batch processing
        embeddings = self.embed_images([img])
        # Return the first (and only) embedding
        return embeddings[0]

    def embed_all(self, imgs):
        """Embed a batch of images.

        Implementation of TorchEmbeddingsMixin.embed_all() method.

        Args:
            imgs: List of images to embed (PIL images or PyTorch tensors)

        Returns:
            numpy array: Embeddings for the images
        """
        # Directly call embed_images which handles batch processing
        return self.embed_images(imgs)

    def get_embeddings(self):
        """Get the last computed embeddings.

        Required override for TorchEmbeddingsMixin to provide embeddings
        in the expected format for FiftyOne.

        Returns:
            numpy array: The last computed embeddings

        Raises:
            ValueError: If no embeddings have been computed yet
        """
        # Check if embeddings capability is enabled
        if not self.has_embeddings:
            raise ValueError("This model instance does not expose embeddings")

        # Check if embeddings have been computed
        if self._last_computed_embeddings is None:
            raise ValueError("No embeddings have been computed yet")

        # Return the stored embeddings as a CPU numpy array
        return self._last_computed_embeddings.detach().cpu().numpy()

    def _get_class_logits(self, text_features, image_features):
        """Calculate similarity scores using SigLIP2's native approach.

        SigLIP2 uses unnormalized embeddings with sigmoid loss, not normalized
        embeddings with softmax like CLIP.

        Args:
            text_features: Text embeddings (numpy array or tensor)
            image_features: Image embeddings (numpy array or tensor)

        Returns:
            tuple: (logits_per_image, logits_per_text) scaled similarity matrices
        """
        with torch.no_grad():
            # Convert numpy arrays to torch tensors and move to device
            if isinstance(text_features, np.ndarray):
                text_features = torch.from_numpy(text_features)
            if isinstance(image_features, np.ndarray):
                image_features = torch.from_numpy(image_features)

            # Ensure correct device and dtype (MPS doesn't support float64)
            # Use .to(device, dtype=...) to cast during move
            text_features = text_features.to(self.model.device, dtype=torch.float32)
            image_features = image_features.to(self.model.device, dtype=torch.float32)

            # NO NORMALIZATION - SigLIP2 embeddings are used as-is

            # Get the logit scale from the model
            logit_scale = self.model.logit_scale.exp()

            # Compute scaled dot product similarity
            logits_per_image = logit_scale * (image_features @ text_features.t())
            logits_per_text = logits_per_image.t()

            return logits_per_image, logits_per_text

    def _predict_all(self, imgs):
        """Run prediction on a batch of images.

        Used for zero-shot classification by comparing image embeddings
        to text embeddings of class names.

        Args:
            imgs: List of images to classify

        Returns:
            numpy array: Raw logits (or processed by output_processor if available)
        """
        # Get image embeddings (returns numpy array)
        image_embeddings = self.embed_images(imgs)

        # Get text embeddings for classes (returns numpy array)
        text_features = self._get_text_features()

        # Calculate similarity between images and text (returns torch tensors)
        logits_per_image, _logits_per_text = self._get_class_logits(
            text_features, image_embeddings
        )

        # If you have an output processor (like CLIP), use it
        if hasattr(self, "_output_processor") and self._output_processor is not None:
            # Get frame size for output processor
            if isinstance(imgs[0], torch.Tensor):
                height, width = imgs[0].size()[-2:]
            elif hasattr(imgs[0], "size"):  # PIL Image
                width, height = imgs[0].size
            else:
                height, width = imgs[0].shape[:2]  # numpy array

            frame_size = (width, height)

            if self.has_logits:
                self._output_processor.store_logits = self.store_logits

            return self._output_processor(
                logits_per_image,
                frame_size,
                confidence_thresh=self.config.confidence_thresh,
            )

        # Return raw logits as CPU numpy array
        return logits_per_image.detach().cpu().numpy()
