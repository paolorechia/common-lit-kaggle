import tasks
from framework import Pipeline


class BasicPredictRandomForestPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            "basic_predict_random_forest",
            [
                tasks.ReadTrainDataTask(),
                tasks.AddBasicFeaturesTrainTask(),
                tasks.TrainBasicRandomForestTask(),
                tasks.ReadPredictionInputDataTask(),
                tasks.AddBasicFeaturesPredictionTask(),
                tasks.PredictBasicRandomForestTask(),
                tasks.WritePredictionsTask(),
            ],
        )
