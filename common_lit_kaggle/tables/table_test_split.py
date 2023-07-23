import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import TrainTestSplitSchema
from common_lit_kaggle.settings import config


class TestSplitTable(TableReference):
    def __init__(self):
        super().__init__(
            name="test_split",
            path=pathlib.Path(config.DATA_TEST_DIR, "test_split.csv"),
            schema=TrainTestSplitSchema,
            format="csv",
        )
