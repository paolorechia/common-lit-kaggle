from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.tasks import (
    ExploreInputDataTask,
    JoinInputTask,
    ReadInputDataTask,
    ReadInputPromptDataTask,
)


class ExploreDataPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            "explore_input_data",
            [
                ReadInputDataTask(),
                ReadInputPromptDataTask(),
                JoinInputTask(),
                ExploreInputDataTask(),
            ],
        )
