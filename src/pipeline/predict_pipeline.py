from __future__ import annotations

import joblib
import pandas as pd


class PredictPipeline:
    def __init__(self, preprocessor_path: str, model_path: str, drop_columns=None):
        self.preprocessor = joblib.load(preprocessor_path)
        self.model = joblib.load(model_path)
        self.drop_columns = drop_columns or []

    def predict(self, input_df: pd.DataFrame):
        features = input_df.drop(columns=self.drop_columns, errors="ignore")
        transformed = self.preprocessor.transform(features)
        return self.model.predict(transformed)
