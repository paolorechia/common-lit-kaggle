import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import UnifiedTextDataSchema
from common_lit_kaggle.settings.config import Config


class UnifiedTextDataTable(TableReference):
    def __init__(self):
        config = Config.get()
        super().__init__(
            name="unified_text_data",
            path=pathlib.Path(config.data_train_dir, "unified_text_data.csv"),
            schema=UnifiedTextDataSchema,
            format="csv",
        )
