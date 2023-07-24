from common_lit_kaggle import tasks
from common_lit_kaggle.framework import Pipeline

label = "set_train_random_forest"

class SentenceTransformerRandomForestPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            label,
            [
                # Train
                tasks.ReadTrainDataTask(),
                tasks.AddBasicFeaturesTrainTask(),
                tasks.AddSentenceEmbeddingToTrainTask(),
                tasks.TrainBasicRandomForestTask(),
                # Test
                tasks.ReadTestDataTask(),
                tasks.AddBasicFeaturesTestTask(),
                tasks.AddSentenceEmbeddingToTestTask(),
                tasks.TestBasicRandomForestTask(name=label),
            ],
        )
