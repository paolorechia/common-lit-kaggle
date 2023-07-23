import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import InputPromptSchema
from common_lit_kaggle.settings import config


class InputPromptsTable(TableReference):
    def __init__(self):
        super().__init__(
            name="input_prompts",
            path=pathlib.Path(config.DATA_INPUT_DIR, "prompts_train.csv"),
            schema=InputPromptSchema,
            format="csv",
        )
