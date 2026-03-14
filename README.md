# Zalo Photos Assignment 2026

This repository contains the solution for the Zalo Photos image tag classification assignment. The approach prioritizes a data-centric mindset, handling real-world noisy labels, severe class imbalance, and semantic ambiguity.

## TL;DR

* Main notebook: [notebooks/main.ipynb](./notebooks/main.ipynb).
* MLflow tracking view: [DagsHub MLflow](https://dagshub.com/cykanp/zalo-photos-assignment-2026.mlflow/#/experiments/1/runs?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D)
* Model definition: [train.py](./src/z_photos/train.py)
* Evaluation notebook: [notebooks/experiment-04-evaluate.ipynb](./notebooks/experiment-04-evaluate.ipynb)

## Setup

1. Install `mise`: This project relies on [mise](https://mise.jdx.dev/) for toolchain and environment management. Please follow the link to install it.
2. We use `uv` for python dependency mapping and execution:

```bash
uv run <command>
```

## Key Files & Workflow

* **notebooks/main.ipynb**: The main notebook containing the data pipeline, EDA insights, and processing logic.
* `src/z_photos/train.py`: The core script used for training the models.
* `notebooks/experiment-04-evaluate.ipynb`: The final evaluation notebook displaying post-training results, performance metrics, and error analysis.

## Data Directory Structure

The project follows a standard medallion architecture for data refinement:

* `data/bronze/`: Raw, original datasets provided by the assignment.
* `data/silver/`: Cleaned and preprocessed data, including duplicate removal and initial noise scoring.
* `data/gold/`: Finalized, fully annotated data ready for training and evaluation holdouts.
* `data/z_photos-gold-train-export/`: The exported `FiftyOne` dataset used directly for model ingestion.

## Tooling & Observability

* FiftyOne: Used extensively for Exploratory Data Analysis (EDA), embedding visualization, and semantic data curation.
* [MLflow tracking view](https://dagshub.com/cykanp/zalo-photos-assignment-2026.mlflow/#/experiments/1/runs?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D)

## AI Usage & Large Assets

* `lfs/`: This directory contains generated objects and large assets such as model checkpoints.
* Note: I utilized AI assistants (LLMs) to construct this project's foundational logic, code structure, and reports, which are also stored among the artifacts here. **Of course under my review and modification.**
