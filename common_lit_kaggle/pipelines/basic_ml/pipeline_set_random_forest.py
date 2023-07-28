from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.tasks import basic_ml, data_split

LABEL = "set_train_random_forest"


class SentenceTransformerRandomForestPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            LABEL,
            [
                # Train
                data_split.ReadTrainDataTask(),
                basic_ml.AddBasicFeaturesTrainTask(),
                basic_ml.AddSentenceEmbeddingToTrainTask(),
                basic_ml.TrainBasicRandomForestTask(),
                # Test
                data_split.ReadTestDataTask(),
                basic_ml.AddBasicFeaturesTestTask(),
                basic_ml.AddSentenceEmbeddingToTestTask(),
                basic_ml.TestBasicRandomForestTask(),
                basic_ml.AnalysePredictionsTask(name=LABEL),
            ],
        )
