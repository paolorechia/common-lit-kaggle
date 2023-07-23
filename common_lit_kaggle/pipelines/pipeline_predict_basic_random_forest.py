from common_lit_kaggle import tasks
from common_lit_kaggle.framework import Pipeline


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
