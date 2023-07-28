from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.tasks import data_split


class SplitTrainTestPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            "split_train_test",
            [
                data_split.ReadInputDataTask(),
                data_split.ReadInputPromptDataTask(),
                data_split.JoinInputTask(),
                data_split.SplitTrainTestByPromptTask(),
            ],
        )
