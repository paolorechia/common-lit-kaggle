from typing import Any, Mapping

from framework import table_io
from framework.task import Task
from tables import TestSplitTable


class ReadTestDataTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = table_io.read_table(TestSplitTable())
        return {"test_data": input_data}
