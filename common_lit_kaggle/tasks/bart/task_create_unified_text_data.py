from typing import Any, Mapping

import polars as pl

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tables import UnifiedTextDataTable


def add_unified_data(train_data: pl.DataFrame) -> pl.DataFrame:
    # Prepare a single string to train transformers
    config = Config.get()
    if config.use_unified_text:
        unified_text_data = train_data.with_columns(
            pl.concat_str(
                [
                    pl.lit("TOPIC TITLE: "),
                    pl.col("prompt_title"),
                    pl.lit("\nREFERENCE TEXT: "),
                    pl.col("prompt_text"),
                    pl.lit("\nQUESTION: \n"),
                    pl.col("prompt_question"),
                    pl.lit("\nSTUDENT ANSWER: \n"),
                    pl.col("text"),
                ]
            ).alias("unified_text")
        )
    else:
        unified_text_data = train_data.with_columns(
            pl.col("text").alias("unified_text")
        )

    return unified_text_data


class CreateUnifiedTextTrainDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        train_data: pl.DataFrame = context["train_data"]
        unified_text_data = add_unified_data(train_data)

        table_io.write_table(
            unified_text_data.select(
                "student_id",
                "prompt_id",
                "content",
                "wording",
                "unified_text",
            ),
            UnifiedTextDataTable(),
        )

        return {"train_unified_text_data": unified_text_data}


class CreateUnifiedTextTestDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        test_data: pl.DataFrame = context["test_data"]
        unified_text_data = add_unified_data(test_data)

        return {"test_unified_text_data": unified_text_data}


class CreateUnifiedTextEvalDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        test_data: pl.DataFrame = context["eval_data"]
        unified_text_data = add_unified_data(test_data)

        return {"eval_unified_text_data": unified_text_data}


class CreateUnifiedTextPredictionDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        prediction_input_data: pl.DataFrame = context["input_prediction_data"]
        unified_text_data = add_unified_data(prediction_input_data)

        return {"prediction_unified_text_data": unified_text_data}
