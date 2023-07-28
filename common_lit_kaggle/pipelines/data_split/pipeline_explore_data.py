from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.tasks import basic_ml, data_split


class ExploreDataPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            "explore_input_data",
            [
                data_split.ReadInputDataTask(),
                data_split.ReadInputPromptDataTask(),
                data_split.JoinInputTask(),
                basic_ml.ExploreInputDataTask(),
            ],
        )
