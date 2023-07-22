from typing import Any, Mapping, Optional

import polars as pl

from framework.task import Task


class ExploreInputDataTask(Task):
    def run(self, context):
        input_data: pl.DataFrame = context["input_data"]
        print(input_data)

        print(input_data.describe())
