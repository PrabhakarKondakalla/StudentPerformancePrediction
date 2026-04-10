import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline


def main():
    samples = pd.DataFrame(
        [
            {
                "gender": "female",
                "race/ethnicity": "group C",
                "parental level of education": "bachelor's degree",
                "lunch": "standard",
                "test preparation course": "completed",
                "reading score": 88,
                "writing score": 90,
            },
            {
                "gender": "male",
                "race/ethnicity": "group D",
                "parental level of education": "some college",
                "lunch": "free/reduced",
                "test preparation course": "Half Completed",
                "reading score": 95,
                "writing score": 89,
            },
            # {
            #     "gender": "female",
            #     "race/ethnicity": "group B",
            #     "parental level of education": "master's degree",
            #     "lunch": "standard",
            #     "test preparation course": "completed",
            #     "reading score": 96,
            #     "writing score": 95,
            # },
        ]
    )

    predictor = PredictPipeline(
        preprocessor_path="artifacts/preprocessor.joblib",
        model_path="artifacts/model.joblib",
    )
    predictions = predictor.predict(samples)

    output = samples.copy()
    output["predicted math score"] = predictions.round(2)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
