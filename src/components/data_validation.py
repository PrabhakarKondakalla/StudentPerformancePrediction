from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DataValidationReport:
    is_valid: bool
    missing_columns: list[str]
    duplicate_rows: int
    null_counts: dict[str, int]


class DataValidation:
    def __init__(self, expected_columns: list[str], target_column: str):
        self.expected_columns = expected_columns
        self.target_column = target_column

    def validate(self, dataframe: pd.DataFrame) -> DataValidationReport:
        missing_columns = [
            column for column in self.expected_columns if column not in dataframe.columns
        ]
        duplicate_rows = int(dataframe.duplicated().sum())
        null_counts = {
            column: int(count)
            for column, count in dataframe.isnull().sum().items()
            if count > 0
        }
        is_valid = not missing_columns and self.target_column in dataframe.columns

        return DataValidationReport(
            is_valid=is_valid,
            missing_columns=missing_columns,
            duplicate_rows=duplicate_rows,
            null_counts=null_counts,
        )
