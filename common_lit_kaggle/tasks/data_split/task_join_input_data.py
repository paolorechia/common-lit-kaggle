from typing import Any, Mapping

import polars as pl

from common_lit_kaggle.framework.task import Task


class JoinInputTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data: pl.DataFrame = context["input_data"]
        prompt_data: pl.DataFrame = context["input_prompts"]

        joined_data = input_data.join(prompt_data, on="prompt_id", how="inner")
        return {"joined_data": joined_data}
