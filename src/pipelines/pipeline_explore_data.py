from framework import Pipeline
from tasks import ExploreInputDataTask


class ExploreDataPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__("explore_input_data", [ExploreInputDataTask()])
