from typing import Any, Mapping

from framework import table_io
from framework.task import Task
from tables import TrainSplitTable


class ReadTrainDataTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = table_io.read_table(TrainSplitTable())
        return {"train_data": input_data}
