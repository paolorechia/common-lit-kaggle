from common_lit_kaggle import tasks
from common_lit_kaggle.framework import Pipeline

LABEL = "basic_train_random_forest"


class BasicRandomForestPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            LABEL,
            [
                tasks.ReadTrainDataTask(),
                tasks.AddBasicFeaturesTrainTask(),
                tasks.TrainBasicRandomForestTask(),
                tasks.ReadTestDataTask(),
                tasks.AddBasicFeaturesTestTask(),
                tasks.TestBasicRandomForestTask(),
                tasks.WritePredictionsTask(),
                tasks.AnalysePredictionsTask(name=LABEL),
            ],
        )
