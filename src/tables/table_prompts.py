import pathlib

from framework.table import TableReference
from schemas.schema_input_prompts import InputPromptSchema
from settings import config


class InputPromptsTable(TableReference):
    def __init__(self):
        super().__init__(
            name="input_prompts",
            path=pathlib.Path(config.DATA_INPUT_DIR, "prompts_train.csv"),
            schema=InputPromptSchema,
            format="csv",
        )
