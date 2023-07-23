import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import InputSummarySchema
from common_lit_kaggle.settings.config import Config


class InputSummariesTable(TableReference):
    def __init__(self):
        config = Config.get()

        super().__init__(
            name="input_summaries",
            path=pathlib.Path(config.data_input_dir, "summaries_train.csv"),
            schema=InputSummarySchema,
            format="csv",
        )
