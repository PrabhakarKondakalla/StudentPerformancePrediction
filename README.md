# Student Performance Prediction with XGBoost

This project is structured as an end-to-end machine learning pipeline for predicting student performance using XGBoost.

## Active Dataset

This project is currently configured for:

- File: `data/raw/StudentsPerformance.csv`
- Dataset style: exam scores with demographic and preparation features
- Target column: `math score`
- Problem type: regression
- Model: `XGBRegressor`

Input features used by default:

- `gender`
- `race/ethnicity`
- `parental level of education`
- `lunch`
- `test preparation course`
- `reading score`
- `writing score`

## Project Structure

```text
StudentPerformancePrediction/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── config.yaml
│   └── schema.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── artifacts/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── logger.py
│   ├── exception.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   └── utils/
│       ├── __init__.py
│       └── common.py
└── tests/
    └── __init__.py
```

## Typical Workflow

1. Put `StudentsPerformance.csv` in `data/raw/`
2. Validate schema and null handling
3. Transform categorical and numerical features
4. Train XGBoost regression model for `math score`
5. Evaluate model performance
6. Save artifacts in `artifacts/`
7. Use prediction pipeline for inference

## Run Training

```bash
.venv/bin/python main.py
```

Training outputs:

- `artifacts/model.joblib`
- `artifacts/preprocessor.joblib`
- `artifacts/metrics.json`
<<<<<<< HEAD
Updating 
=======

>>>>>>> 655f75a (Chnaged to previous Update)
