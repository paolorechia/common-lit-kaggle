import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import OutputPredictionSchema
from common_lit_kaggle.settings.config import Config


class OutputPredictionTable(TableReference):
    def __init__(self):
        config = Config.get()

        super().__init__(
            name="output_prediction",
            path=pathlib.Path(config.data_output_dir, "submission.csv"),
            schema=OutputPredictionSchema,
            format="csv",
        )
