from typing import Any, Mapping

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.tables import TestSplitTable


class ReadTestDataTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = table_io.read_table(TestSplitTable())
        return {"test_data": input_data}
