from typing import Any, Mapping

from framework import table_io
from framework.task import Task
from tables.table_summaries import InputSummaries


class ReadInputDataTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = table_io.read_table(InputSummaries())
        return {"input_data": input_data}
