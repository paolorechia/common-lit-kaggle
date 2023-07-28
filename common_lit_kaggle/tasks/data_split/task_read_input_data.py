from typing import Any, Mapping

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.tables.table_summaries import InputSummariesTable


class ReadInputDataTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = table_io.read_table(InputSummariesTable())
        return {"input_data": input_data}
