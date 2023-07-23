import pathlib

from framework.table import TableReference
from schemas import TextsPerPromptSchema
from settings import config


class TextsPerPromptTable(TableReference):
    def __init__(self):
        super().__init__(
            name="texts_per_prompt",
            path=pathlib.Path(config.DATA_EXPLORATION_DIR, "texts_per_prompt.csv"),
            schema=TextsPerPromptSchema,
            format="csv",
        )
