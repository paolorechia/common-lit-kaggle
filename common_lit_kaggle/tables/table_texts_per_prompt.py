import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import TextsPerPromptSchema
from common_lit_kaggle.settings.config import Config


class TextsPerPromptTable(TableReference):
    def __init__(self):
        config = Config.get()

        super().__init__(
            name="texts_per_prompt",
            path=pathlib.Path(config.data_exploration_dir, "texts_per_prompt.csv"),
            schema=TextsPerPromptSchema,
            format="csv",
        )
