import pathlib

from framework.table import TableReference
from schemas import InputSummarySchema
from settings import config

INPUT_CSV = "summaries_train.csv"
INPUT_CSV_FULL_PATH = pathlib.Path(config.DATA_INPUT_DIR, INPUT_CSV)


class InputSummariesTable(TableReference):
    def __init__(self):
        super().__init__(
            name="input_summaries",
            path=INPUT_CSV_FULL_PATH,
            schema=InputSummarySchema,
            format="csv",
        )
