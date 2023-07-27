from typing import Any, Mapping

import polars as pl

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.tables import UnifiedTextDataTable


class CreateUnifiedTextTrainDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        train_data: pl.DataFrame = context["train_data"]

        # Prepare a single string to train transformers
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

        # If we want to train a GPT-like model, we can include the labels as part of the text
        # Have the string ready, so later we can just concatenate together with the unified_text
        unified_text_data = unified_text_data.with_columns(
            pl.concat_str(
                [
                    pl.lit("\n\nGRADING SECTION"),
                    pl.lit("\nWORDING: "),
                    pl.col("wording"),
                    pl.lit("\nCONTENT: "),
                    pl.col("content"),
                    pl.lit("\nEND_OF_TEXT\n"),
                ]
            ).alias("unified_labels")
        )

        table_io.write_table(
            unified_text_data.select(
                "student_id",
                "prompt_id",
                "content",
                "wording",
                "unified_text",
                "unified_labels",
            ),
            UnifiedTextDataTable(),
        )

        return {"train_unified_text_data": unified_text_data}


class CreateUnifiedTextTestDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        train_data: pl.DataFrame = context["test_data"]

        # Prepare a single string to train transformers
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

        # If we want to train a GPT-like model, we can include the labels as part of the text
        # Have the string ready, so later we can just concatenate together with the unified_text
        unified_text_data = unified_text_data.with_columns(
            pl.concat_str(
                [
                    pl.lit("\n\nGRADING SECTION"),
                    pl.lit("\nWORDING: "),
                    pl.col("wording"),
                    pl.lit("\nCONTENT: "),
                    pl.col("content"),
                    pl.lit("\nEND_OF_TEXT\n"),
                ]
            ).alias("unified_labels")
        )

        table_io.write_table(
            unified_text_data.select(
                "student_id",
                "prompt_id",
                "content",
                "wording",
                "unified_text",
                "unified_labels",
            ),
            UnifiedTextDataTable(),
        )

        return {"test_unified_text_data": unified_text_data}
