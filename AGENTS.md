# AGENTS.md

## Scope

This repo is for a 3-day take-home on image tag classification with noisy, ambiguous, and imbalanced real-world data.

## Environment

- Use `uv` for environment and package management. Run `uv run ruff check --fix .` and `uv run ruff format .` after Python changes.
- Use modern Python typing (`dict[str, int]`, `X | None`).
- Prefer small, reversible changes. Do not reorganize the repo unless needed.
- Prefer installed skills when relevant (e.g. FiftyOne for EDA, MLflow for tracking).
- Keep reusable logic in modules/scripts; keep notebooks minimal and reproducible.

## Task Constraints

- Treat this as single-label classification unless explicitly asked otherwise.
- Prioritize reproducibility, clear evaluation, and concise reporting.
