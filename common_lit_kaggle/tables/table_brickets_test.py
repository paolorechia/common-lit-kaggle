import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import TrainTestSplitSchema
from common_lit_kaggle.settings.config import Config


class BricketsTestTable(TableReference):
    def __init__(self):
        config = Config.get()
        super().__init__(
            name="brickets_test",
            path=pathlib.Path(config.data_train_dir, "brickets_test.csv"),
            schema=TrainTestSplitSchema,
            format="csv",
        )
