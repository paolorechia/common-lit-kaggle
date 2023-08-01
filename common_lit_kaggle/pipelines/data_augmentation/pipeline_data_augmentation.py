import json

from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tasks import data_balancing, data_split


class AugmentWord2VecTrainDataPipeline(Pipeline):
    def __init__(self) -> None:
        config = Config.get()

        super().__init__(
            "augment_word2vec",
            [
                # Load training data
                data_split.ReadTrainDataTask(),
                data_balancing.BucketTrainDataTask(),
                data_balancing.AugmentWord2VecTrainDataTask(),
            ],
        )
