from typing import Any, Mapping

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.tables import InputPredictionPromptsTable


class ReadPredictionInputPromptDataTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_prompts = table_io.read_table(InputPredictionPromptsTable())
        return {"input_prediction_prompts": input_prompts}
