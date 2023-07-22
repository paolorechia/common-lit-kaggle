from framework import Pipeline
from tasks import (
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
