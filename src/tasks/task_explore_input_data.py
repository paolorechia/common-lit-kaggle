from typing import Any, Mapping

import polars as pl

from framework.task import Task


class ExploreInputDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data: pl.DataFrame = context["input_data"]
        print(input_data)

        print(input_data.describe())
        return {}
