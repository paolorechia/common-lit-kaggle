import pathlib

from framework.table import TableReference
from schemas import OutputPredictionSchema
from settings import config


class OutputPredictionTable(TableReference):
    def __init__(self):
        super().__init__(
            name="output_prediction",
            path=pathlib.Path(config.DATA_OUTPUT_DIR, "submission.csv"),
            schema=OutputPredictionSchema,
            format="csv",
        )
