import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import StudentsPerTextSchema
from common_lit_kaggle.settings import config


class StudentsPerTextTable(TableReference):
    def __init__(self):
        super().__init__(
            name="students_per_text",
            path=pathlib.Path(config.DATA_EXPLORATION_DIR, "students_per_text.csv"),
            schema=StudentsPerTextSchema,
            format="csv",
        )
