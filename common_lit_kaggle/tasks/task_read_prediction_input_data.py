from typing import Any, Mapping

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.tables import (
    InputPredictionPromptsTable,
    InputPredictionSummariesTable,
)


class ReadPredictionInputDataTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = table_io.read_table(InputPredictionSummariesTable())
        prompts = table_io.read_table(InputPredictionPromptsTable())
        joined_data = input_data.join(prompts, on="prompt_id", how="inner")
        return {"input_prediction_data": joined_data}
