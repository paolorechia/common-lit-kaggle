from typing import Any, Mapping

import polars as pl

from framework import table_io
from framework.task import Task
from tables import TestSplitTable, TrainSplitTable


class SplitTrainTestByPromptTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data: pl.DataFrame = context["joined_data"]

        train_prompts = [
            "3b9047",
            "39c16e",
        ]
        test_prompts = [
            "ebad26",
            "814d6b",
        ]

        train_data = input_data.filter(pl.col("prompt_id").is_in(train_prompts))

        assert sorted(train_prompts) == sorted(
            train_data.select("prompt_id").unique().to_series().to_list()
        )

        test_data = input_data.filter(pl.col("prompt_id").is_in(test_prompts))

        assert sorted(test_prompts) == sorted(
            test_data.select("prompt_id").unique().to_series().to_list()
        )

        table_io.write_table(train_data, TrainSplitTable())
        table_io.write_table(test_data, TestSplitTable())

        return {
            "train_split": train_data,
            "test_data": test_data,
        }
