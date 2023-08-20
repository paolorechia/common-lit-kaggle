import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import SyntheticDataSchema
from common_lit_kaggle.settings.config import Config


class RLGPT2SyntheticData(TableReference):
    def __init__(self):
        config = Config.get()
        super().__init__(
            name="train_rl_gpt2",
            path=pathlib.Path(config.data_train_dir, "train_rl_gpt2.csv"),
            schema=SyntheticDataSchema,
            format="csv",
        )
