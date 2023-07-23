import pathlib

from framework.table import TableReference
from schemas import InputPromptSchema
from settings import config


class InputPredictionPromptsTable(TableReference):
    def __init__(self):
        super().__init__(
            name="input_prediction_prompts",
            path=pathlib.Path(config.DATA_INPUT_DIR, "prompts_test.csv"),
            schema=InputPromptSchema,
            format="csv",
        )
