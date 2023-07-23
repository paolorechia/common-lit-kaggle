import pathlib

from framework.table import TableReference
from schemas import StudentsPerTextSchema
from settings import config


class StudentsPerTextTable(TableReference):
    def __init__(self):
        super().__init__(
            name="students_per_text",
            path=pathlib.Path(config.DATA_EXPLORATION_DIR, "students_per_text.csv"),
            schema=StudentsPerTextSchema,
            format="csv",
        )
