# Chat History Reference

## Project Goal

Build an end-to-end Student Performance Prediction project using XGBoost.

## Folder Structure Created

The project was scaffolded with these main directories:

- `config/`
- `data/raw/`
- `data/interim/`
- `data/processed/`
- `artifacts/`
- `notebooks/`
- `src/components/`
- `src/pipeline/`
- `src/utils/`
- `tests/`

## Dataset Discussion

Several dataset options were discussed. The recommended primary dataset was:

- UCI Student Performance dataset
- Official page: `https://archive.ics.uci.edu/dataset/320/student+performance`
- Common CSV reference used during setup: `student-mat.csv`

Kaggle mirrors were also identified, including:

- `https://www.kaggle.com/datasets/uciml/student-alcohol-consumption/data`
- `https://www.kaggle.com/datasets/imkrkannan/student-performance-data-set-y-uci`

## Target Definition

The project is now configured for the dataset present in the workspace:

- Dataset file: `data/raw/StudentsPerformance.csv`
- Target column: `math score`
- Problem type: regression
- Model: `XGBRegressor`

## Current Dataset Added By User

The dataset currently present in the workspace is:

- `data/raw/StudentsPerformance.csv`

Detected columns:

- `gender`
- `race/ethnicity`
- `parental level of education`
- `lunch`
- `test preparation course`
- `math score`
- `reading score`
- `writing score`

## Active Schema

The active pipeline now matches `StudentsPerformance.csv`.

Features:

- `gender`
- `race/ethnicity`
- `parental level of education`
- `lunch`
- `test preparation course`
- `reading score`
- `writing score`

Target:

- `math score`

## Current Code Status

The following were already built:

- end-to-end project structure
- data ingestion component
- data validation component
- data transformation component
- XGBoost model trainer
- evaluation logic
- training entry point in `main.py`

## Recommended Next Step

Run training on the active dataset:

```bash
.venv/bin/python main.py
```

## Decision Snapshot

As of now:

- codebase target is `math score`
- actual dataset in `data/raw/` is `StudentsPerformance.csv`
- config and schema are aligned to the dataset currently in the workspace
