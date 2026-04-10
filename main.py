from src.pipeline.train_pipeline import run_training_pipeline


if __name__ == "__main__":
    result = run_training_pipeline()
    metrics = result["metrics"]

    print(f"Dataset: {result['dataset']}")
    print(f"Target Column: {result['target_column']}")
    print(f"Problem Type: {result['problem_type']}")
    print(f"Duplicate Rows: {result['validation']['duplicate_rows']}")
    print(f"Null Counts: {result['validation']['null_counts']}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2_Score: {metrics['r2_score']:.4f}")
