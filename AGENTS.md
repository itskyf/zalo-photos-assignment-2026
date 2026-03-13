# AGENTS.md

## Overview

This repo is for a 3-day take-home on image tag classification with noisy, ambiguous, and imbalanced real-world data.

## Build and test commands

- Python: `uv run ...`

## Mandatory skill usage

- `fiftyone-*` for EDA and evaluation.
- `mlflow` for tracking.
- `./skills/jupyter-notebook` for Jupyter notebooks.

## Working Rules

- Use Google-style docstrings. Keep comments, docstrings, and logging minimal, clear, and strictly necessary.
- Use modern Python typing (`dict[str, int]`, `X | None`); avoid `Any` and hand-crafted attribute names.
- Make small, reversible changes; avoid repo-wide reorganization unless necessary.
- Keep reusable logic in modules/scripts; keep notebooks minimal and reproducible.
- Prioritize reproducibility, clear evaluation, and concise reporting.
