from framework import Pipeline
from tasks import AddBasicFeaturesTask, ReadTrainDataTask, TrainBasicRandomForestTask


class BasicRandomForestPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            "basic_random_forest",
            [ReadTrainDataTask(), AddBasicFeaturesTask(), TrainBasicRandomForestTask()],
        )
