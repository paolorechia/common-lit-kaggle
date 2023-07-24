from common_lit_kaggle import tasks
from common_lit_kaggle.framework import Pipeline


label = "basic_train_random_forest"

class BasicRandomForestPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            label,
            [
                tasks.ReadTrainDataTask(),
                tasks.AddBasicFeaturesTrainTask(),
                tasks.TrainBasicRandomForestTask(),
                tasks.ReadTestDataTask(),
                tasks.AddBasicFeaturesTestTask(),
                tasks.TestBasicRandomForestTask(name=label),
            ],
        )
