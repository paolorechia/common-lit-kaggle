import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import StudentsPerTextSchema
from common_lit_kaggle.settings.config import Config


class StudentsPerTextTable(TableReference):
    def __init__(self):
        config = Config.get()

        super().__init__(
            name="students_per_text",
            path=pathlib.Path(config.data_exploration_dir, "students_per_text.csv"),
            schema=StudentsPerTextSchema,
            format="csv",
        )
