from typing import Any, Mapping

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.tables import BricketsTestTable


class ReadBricketsTestTableTask(Task):
    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)

    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        train_data = table_io.read_table(BricketsTestTable())

        print(train_data)
        # Replace train data in pipeline with samples through
        # Brickets method
        return {"train_data": train_data}
