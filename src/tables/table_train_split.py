import pathlib

from framework.table import TableReference
from schemas import TrainTestSplitSchema
from settings import config


class TrainSplitTable(TableReference):
    def __init__(self):
        super().__init__(
            name="train_split",
            path=pathlib.Path(config.DATA_TRAIN_DIR, "train_split.csv"),
            schema=TrainTestSplitSchema,
            format="csv",
        )
