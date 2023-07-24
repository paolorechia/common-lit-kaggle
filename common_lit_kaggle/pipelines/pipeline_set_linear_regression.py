from common_lit_kaggle import tasks
from common_lit_kaggle.framework import Pipeline

LABEL = "set_train_linear"


class SentenceTransformerLinearRegressionPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            LABEL,
            [
                # Train
                tasks.ReadTrainDataTask(),
                tasks.AddBasicFeaturesTrainTask(),
                tasks.AddSentenceEmbeddingToTrainTask(),
                tasks.TrainBasicLinearRegressorTask(),
                # Test
                tasks.ReadTestDataTask(),
                tasks.AddBasicFeaturesTestTask(),
                tasks.AddSentenceEmbeddingToTestTask(),
                tasks.TestBasicLinearRegressorTask(),
                tasks.AnalysePredictionsTask(name=LABEL),
            ],
        )
