from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from framework.task import Task


class ExploreInputDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data: pl.DataFrame = context["joined_data"]

        print(input_data)
        texts_per_student = input_data.groupby("student_id").agg(pl.count())

        texts_per_prompt = input_data.groupby("prompt_id").agg(pl.count())

        print(texts_per_student.describe())

        print(texts_per_prompt)

        print(input_data.describe())
        return {}
