from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.tasks import basic_ml, data_split

LABEL = "set_train_linear"


class SentenceTransformerLinearRegressionPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            LABEL,
            [
                # Train
                data_split.ReadTrainDataTask(),
                basic_ml.AddBasicFeaturesTrainTask(),
                basic_ml.AddSentenceEmbeddingToTrainTask(),
                basic_ml.TrainBasicLinearRegressorTask(),
                # Test
                data_split.ReadTestDataTask(),
                basic_ml.AddBasicFeaturesTestTask(),
                basic_ml.AddSentenceEmbeddingToTestTask(),
                basic_ml.TestBasicLinearRegressorTask(),
                basic_ml.AnalysePredictionsTask(name=LABEL),
            ],
        )
