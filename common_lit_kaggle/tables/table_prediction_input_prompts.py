import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import InputPromptSchema
from common_lit_kaggle.settings.config import Config


class InputPredictionPromptsTable(TableReference):
    def __init__(self):
        config = Config.get()

        super().__init__(
            name="input_prediction_prompts",
            path=pathlib.Path(config.data_input_dir, "prompts_test.csv"),
            schema=InputPromptSchema,
            format="csv",
        )
