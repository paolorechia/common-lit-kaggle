from framework import Pipeline
from tasks import (
    JoinInputTask,
    ReadInputDataTask,
    ReadInputPromptDataTask,
    SplitTrainTestByPromptTask,
)


class SplitTrainTestPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            "split_train_test",
            [
                ReadInputDataTask(),
                ReadInputPromptDataTask(),
                JoinInputTask(),
                SplitTrainTestByPromptTask(),
            ],
        )
