from __future__ import annotations

from dataclasses import dataclass

import joblib
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    model_path: str
    params: dict


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def run(self, X_train, y_train):
        params = {
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": 1,
            **self.config.params,
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        joblib.dump(model, self.config.model_path)
        return model
