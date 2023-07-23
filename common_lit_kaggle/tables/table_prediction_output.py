import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import OutputPredictionSchema
from common_lit_kaggle.settings import config


class OutputPredictionTable(TableReference):
    def __init__(self):
        super().__init__(
            name="output_prediction",
            path=pathlib.Path(config.DATA_OUTPUT_DIR, "submission.csv"),
            schema=OutputPredictionSchema,
            format="csv",
        )
