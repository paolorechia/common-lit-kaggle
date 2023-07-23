import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import TextsPerPromptSchema
from common_lit_kaggle.settings import config


class TextsPerPromptTable(TableReference):
    def __init__(self):
        super().__init__(
            name="texts_per_prompt",
            path=pathlib.Path(config.DATA_EXPLORATION_DIR, "texts_per_prompt.csv"),
            schema=TextsPerPromptSchema,
            format="csv",
        )
