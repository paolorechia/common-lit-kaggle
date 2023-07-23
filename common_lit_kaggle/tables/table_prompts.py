import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import InputPromptSchema
from common_lit_kaggle.settings.config import Config


class InputPromptsTable(TableReference):
    def __init__(self):
        config = Config.get()

        super().__init__(
            name="input_prompts",
            path=pathlib.Path(config.data_input_dir, "prompts_train.csv"),
            schema=InputPromptSchema,
            format="csv",
        )
