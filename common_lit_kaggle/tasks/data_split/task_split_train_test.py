from typing import Any, Mapping

import polars as pl

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tables import EvalSplitTable, TestSplitTable, TrainSplitTable


class SplitTrainTestByPromptTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        config = Config.get()
        input_data: pl.DataFrame = context["joined_data"]

        train_prompts = config.train_prompts
        test_prompts = config.test_prompts
        eval_prompts = config.eval_prompts

        train_data = input_data.filter(pl.col("prompt_id").is_in(train_prompts))

        assert sorted(train_prompts) == sorted(
            train_data.select("prompt_id").unique().to_series().to_list()
        )

        test_data = input_data.filter(pl.col("prompt_id").is_in(test_prompts))

        assert sorted(test_prompts) == sorted(
            test_data.select("prompt_id").unique().to_series().to_list()
        )

        eval_data = input_data.filter(pl.col("prompt_id").is_in(eval_prompts))

        assert sorted(eval_prompts) == sorted(
            eval_data.select("prompt_id").unique().to_series().to_list()
        )

        table_io.write_table(train_data, TrainSplitTable())
        table_io.write_table(test_data, TestSplitTable())
        table_io.write_table(eval_data, EvalSplitTable())

        return {
            "train_split": train_data,
            "test_data": test_data,
            "eval_data": eval_data,
        }
