import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import UnifiedTextDataSchema
from common_lit_kaggle.settings.config import Config


class AugmentedLlamaTrainTable(TableReference):
    def __init__(self):
        config = Config.get()
        super().__init__(
            name="train_llama_augmented",
            path=pathlib.Path(config.data_train_dir, "train_llama_augmented_split.csv"),
            schema=UnifiedTextDataSchema,
            format="csv",
        )
