from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.tasks import (
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
