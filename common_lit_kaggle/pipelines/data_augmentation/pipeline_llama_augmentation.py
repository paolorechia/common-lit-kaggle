import json

from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tasks import bart, data_balancing, data_split


class AugmentLlamaTrainDataPipeline(Pipeline):
    def __init__(self) -> None:
        config = Config.get()

        super().__init__(
            "augment_llama",
            [
                data_split.ReadTrainDataTask(),
                bart.CreateUnifiedTextTrainDataTask(),
                data_balancing.LlamaAugmenterTask(),
            ],
        )
