import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import TrainTestSplitSchema
from common_lit_kaggle.settings.config import Config


class TestSplitTable(TableReference):
    def __init__(self):
        config = Config.get()

        super().__init__(
            name="test_split",
            path=pathlib.Path(config.data_test_dir, "test_split.csv"),
            schema=TrainTestSplitSchema,
            format="csv",
        )
