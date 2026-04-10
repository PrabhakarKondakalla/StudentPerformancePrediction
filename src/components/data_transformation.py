from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DataTransformationConfig:
    target_column: str
    preprocessor_path: str
    drop_columns: list[str]


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def _split_features_and_target(self, dataframe: pd.DataFrame):
        columns_to_drop = [self.config.target_column, *self.config.drop_columns]
        X = dataframe.drop(columns=columns_to_drop, errors="ignore")
        y = dataframe[self.config.target_column]
        return X, y

    def _build_preprocessor(self, X_train: pd.DataFrame) -> ColumnTransformer:
        categorical_columns = X_train.select_dtypes(include=["object"]).columns.tolist()
        numerical_columns = X_train.select_dtypes(exclude=["object"]).columns.tolist()

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("categorical", categorical_pipeline, categorical_columns),
                ("numerical", numerical_pipeline, numerical_columns),
            ]
        )

    def fit_transform(self, train_path: str):
        train_df = pd.read_csv(train_path)
        X_train, y_train = self._split_features_and_target(train_df)
        preprocessor = self._build_preprocessor(X_train)
        X_train_processed = preprocessor.fit_transform(X_train)
        joblib.dump(preprocessor, self.config.preprocessor_path)
        return X_train_processed, y_train

    def transform(self, test_path: str):
        test_df = pd.read_csv(test_path)
        X_test, y_test = self._split_features_and_target(test_df)
        preprocessor = joblib.load(self.config.preprocessor_path)
        X_test_processed = preprocessor.transform(X_test)
        return X_test_processed, y_test
