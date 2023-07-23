from typing import Any, Mapping

from framework import table_io
from framework.task import Task
from tables import InputPredictionPromptsTable


class ReadPredictionInputPromptDataTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_prompts = table_io.read_table(InputPredictionPromptsTable())
        return {"input_prediction_prompts": input_prompts}
