from __future__ import annotations

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluation:
    @staticmethod
    def evaluate(model, X_test, y_test):
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "mse": float(mse),
            "rmse": float(mse ** 0.5),
            "r2_score": float(r2_score(y_test, predictions)),
        }
