from typing import Any, Mapping

from framework import table_io
from framework.task import Task
from tables import InputPredictionSummariesTable


class ReadPredictionInputDataTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = table_io.read_table(InputPredictionSummariesTable())
        return {"input_prediction_data": input_data}
