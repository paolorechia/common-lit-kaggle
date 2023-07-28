from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.tasks import basic_ml, data_split


class SentenceTransformersPredictRandomForestPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            "set_predict_random_forest",
            [
                # Train
                data_split.ReadTrainDataTask(),
                basic_ml.AddBasicFeaturesTrainTask(),
                basic_ml.AddSentenceEmbeddingToTrainTask(),
                basic_ml.TrainBasicRandomForestTask(),
                # Test
                data_split.ReadPredictionInputDataTask(),
                basic_ml.AddBasicFeaturesPredictionTask(),
                basic_ml.AddSentenceEmbeddingToPredictTask(),
                basic_ml.PredictBasicRandomForestTask(),
                data_split.WritePredictionsTask(),
            ],
        )
