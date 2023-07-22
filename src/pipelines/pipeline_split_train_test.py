from framework import Pipeline
from tasks import SplitTrainTestTask


class SplitTrainTestPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__("split_train_test", [SplitTrainTestTask()])
