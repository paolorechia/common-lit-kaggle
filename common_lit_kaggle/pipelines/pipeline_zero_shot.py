from common_lit_kaggle import tasks
from common_lit_kaggle.framework import Pipeline

LABEL = "zero_train_random_forest"


class ZeroShotRandomForestPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            LABEL,
            [
                tasks.ReadTrainDataTask(),
                tasks.AddBasicFeaturesTrainTask(),
                tasks.AddZeroShotLabelTrainTask(),
                tasks.TrainBasicRandomForestTask(),
                tasks.ReadTestDataTask(),
                tasks.AddBasicFeaturesTestTask(),
                tasks.TestBasicRandomForestTask(name=LABEL),
            ],
        )
