import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import AugmentedTrainSplit
from common_lit_kaggle.settings.config import Config


class AugmentedWmt19TrainTable(TableReference):
    def __init__(self):
        config = Config.get()
        super().__init__(
            name="train_wmt19_augmented",
            path=pathlib.Path(config.data_train_dir, "train_wmt19_augmented_split.csv"),
            schema=AugmentedTrainSplit,
            format="csv",
        )
