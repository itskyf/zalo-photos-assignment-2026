"""Module for FiftyOne custom model registrations."""

import os

import fiftyone as fo


def register_models():
    """Register custom models to the FiftyOne Model Zoo."""
    manifest_path = os.path.join(os.path.dirname(__file__), "manifest.json")

    current_paths = fo.config.model_zoo_manifest_paths or []
    if manifest_path not in current_paths:
        fo.config.model_zoo_manifest_paths = [*current_paths, manifest_path]
