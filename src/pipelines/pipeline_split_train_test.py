from framework import Pipeline
from tasks.task_split_train_test import SplitTrainTestTask


class SplitTrainTestPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__([SplitTrainTestTask()])
