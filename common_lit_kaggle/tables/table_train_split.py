import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import TrainTestSplitSchema
from common_lit_kaggle.settings import config


class TrainSplitTable(TableReference):
    def __init__(self):
        super().__init__(
            name="train_split",
            path=pathlib.Path(config.DATA_TRAIN_DIR, "train_split.csv"),
            schema=TrainTestSplitSchema,
            format="csv",
        )
