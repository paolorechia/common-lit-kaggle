import pathlib

from framework.table import TableReference
from schemas import InputSummarySchema
from settings import config


class InputPredictionSummariesTable(TableReference):
    def __init__(self):
        super().__init__(
            name="input_prediction_summaries",
            path=pathlib.Path(config.DATA_INPUT_DIR, "summaries_test.csv"),
            schema=InputSummarySchema,
            format="csv",
        )
