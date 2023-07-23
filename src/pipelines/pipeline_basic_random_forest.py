import tasks
from framework import Pipeline


class BasicRandomForestPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            "basic_train_random_forest",
            [
                tasks.ReadTrainDataTask(),
                tasks.AddBasicFeaturesTrainTask(),
                tasks.TrainBasicRandomForestTask(),
                tasks.ReadTestDataTask(),
                tasks.AddBasicFeaturesTestTask(),
                tasks.TestBasicRandomForestTask(),
            ],
        )
