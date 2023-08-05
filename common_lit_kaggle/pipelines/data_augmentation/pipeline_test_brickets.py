from common_lit_kaggle import tables
from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tasks import data_balancing


class BricketsTestPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            "brickets_test",
            [
                data_balancing.BricketsTask(tables.AugmentedBertTrainTable()),
            ],
        )
