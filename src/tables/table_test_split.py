import pathlib

from framework.table import TableReference
from schemas import TrainTestSplitSchema
from settings import config


class TestSplitTable(TableReference):
    def __init__(self):
        super().__init__(
            name="test_split",
            path=pathlib.Path(config.DATA_TEST_DIR, "test_split.csv"),
            schema=TrainTestSplitSchema,
            format="csv",
        )
