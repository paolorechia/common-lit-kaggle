import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import TrainTestSplitSchema
from common_lit_kaggle.settings.config import Config


class TrainSplitTable(TableReference):
    def __init__(self):
        config = Config.get()
        super().__init__(
            name="train_split",
            path=pathlib.Path(config.data_train_dir, "train_split.csv"),
            schema=TrainTestSplitSchema,
            format="csv",
        )
