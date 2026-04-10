from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.data_validation import DataValidation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.utils.common import ensure_directory, read_yaml


def run_training_pipeline(config_path: str = "config/config.yaml"):
    config = read_yaml(config_path)
    schema = read_yaml("config/schema.yaml")

    ingestion = DataIngestion(
        DataIngestionConfig(
            raw_data_path=config["paths"]["raw_data"],
            train_data_path=config["paths"]["train_data"],
            test_data_path=config["paths"]["test_data"],
            test_size=config["test_size"],
            random_state=config["random_state"],
        )
    )
    raw_path, train_path, test_path = ingestion.run()

    raw_df = pd.read_csv(raw_path)
    validator = DataValidation(
        expected_columns=list(schema["columns"].keys()),
        target_column=config["target_column"],
    )
    validation_report = validator.validate(raw_df)
    if not validation_report.is_valid:
        raise ValueError(
            f"Dataset validation failed. Missing columns: {validation_report.missing_columns}"
        )

    transformation = DataTransformation(
        DataTransformationConfig(
            target_column=config["target_column"],
            preprocessor_path=config["paths"]["preprocessor"],
            drop_columns=config.get("drop_columns", []),
        )
    )
    X_train, y_train = transformation.fit_transform(train_path)
    X_test, y_test = transformation.transform(test_path)

    trainer = ModelTrainer(
        ModelTrainerConfig(
            model_path=config["paths"]["model"],
            params=config["xgboost"],
        )
    )
    model = trainer.run(X_train, y_train)
    metrics = ModelEvaluation.evaluate(model, X_test, y_test)

    metrics_path = Path(config["paths"]["metrics"])
    ensure_directory(metrics_path.parent)
    payload = {
        "dataset": config["paths"]["raw_data"],
        "target_column": config["target_column"],
        "problem_type": config["problem_type"],
        "validation": {
            "duplicate_rows": validation_report.duplicate_rows,
            "null_counts": validation_report.null_counts,
        },
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
