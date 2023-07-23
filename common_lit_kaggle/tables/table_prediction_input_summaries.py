import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import InputSummarySchema
from common_lit_kaggle.settings import config


class InputPredictionSummariesTable(TableReference):
    def __init__(self):
        super().__init__(
            name="input_prediction_summaries",
            path=pathlib.Path(config.DATA_INPUT_DIR, "summaries_test.csv"),
            schema=InputSummarySchema,
            format="csv",
        )
