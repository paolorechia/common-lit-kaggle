from common_lit_kaggle import tasks
from common_lit_kaggle.framework import Pipeline


class SentenceTransformersPredictRandomForestPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            "set_predict_random_forest",
            [
                # Train
                tasks.ReadTrainDataTask(),
                tasks.AddBasicFeaturesTrainTask(),
                tasks.AddSentenceEmbeddingToTrainTask(),
                tasks.TrainBasicRandomForestTask(),
                # Test
                tasks.ReadPredictionInputDataTask(),
                tasks.AddBasicFeaturesPredictionTask(),
                tasks.AddSentenceEmbeddingToPredictTask(),
                tasks.PredictBasicRandomForestTask(),
                tasks.WritePredictionsTask(),
            ],
        )
