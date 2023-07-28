from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.tasks import basic_ml, data_split


class BasicPredictRandomForestPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            "basic_predict_random_forest",
            [
                # Train
                data_split.ReadTrainDataTask(),
                basic_ml.AddBasicFeaturesTrainTask(),
                basic_ml.TrainBasicRandomForestTask(),
                # Predict
                data_split.ReadPredictionInputDataTask(),
                basic_ml.AddBasicFeaturesPredictionTask(),
                basic_ml.PredictBasicRandomForestTask(),
                # Write output
                data_split.WritePredictionsTask(),
            ],
        )
