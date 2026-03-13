---
name: "jupyter-notebook"
description: "Use when the user asks to create, scaffold, or edit Jupyter notebooks (`.ipynb`) for experiments, explorations, or tutorials; prefer the bundled templates and run the helper script `new_notebook.py` to generate a clean starting notebook. Use Jupyter MCP tools to execute Python code statefully."
---

# Jupyter Notebook Skill

Create clean, reproducible Jupyter notebooks for two primary modes:

- Experiments and exploratory analysis
- Tutorials and teaching-oriented walkthroughs

Prefer the bundled templates and the helper script for consistent structure and fewer JSON mistakes.

## When to use

- Create a new `.ipynb` notebook from scratch.
- Convert rough notes or scripts into a structured notebook.
- Refactor an existing notebook to be more reproducible and skimmable.
- Build experiments or tutorials that will be read or re-run by other people.

## Decision tree

- If the request is exploratory, analytical, or hypothesis-driven, choose `experiment`.
- If the request is instructional, step-by-step, or audience-specific, choose `tutorial`.
- If editing an existing notebook, treat it as a refactor: preserve intent and improve structure.

## Workflow

1. Lock the intent.
Identify the notebook kind: `experiment` or `tutorial`.
Capture the objective, audience, and what "done" looks like.

2. Fill the notebook with small, runnable steps.
Keep each code cell focused on one step.
Add short markdown cells that explain the purpose and expected result.
Avoid large, noisy outputs when a short summary works.

3. Apply the right pattern.
For experiments, follow `references/experiment-patterns.md`.
For tutorials, follow `references/tutorial-patterns.md`.

4. Edit safely when working with existing notebooks.
Preserve the notebook structure; avoid reordering cells unless it improves the top-to-bottom story.
Prefer targeted edits over full rewrites.
If you must edit raw JSON, review `references/notebook-structure.md` first.

5. Validate the result.

Use Jupyter MCP tools to execute Python code statefully and run the notebook top-to-bottom.
If execution is not possible, say so explicitly and call out how to validate locally.
Use the final pass checklist in `references/quality-checklist.md`.

## Temp and output conventions

- Use `tmp/jupyter-notebook/` for intermediate files; delete when done.
- Write final artifacts under `output/jupyter-notebook/` when working in this repo.
- Use stable, descriptive filenames (for example, `ablation-temperature.ipynb`).

## Environment

No required environment variables.

## Reference map

- `references/experiment-patterns.md`: experiment structure and heuristics.
- `references/tutorial-patterns.md`: tutorial structure and teaching flow.
- `references/notebook-structure.md`: notebook JSON shape and safe editing rules.
- `references/quality-checklist.md`: final validation checklist.
