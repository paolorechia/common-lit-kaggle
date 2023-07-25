from typing import Any, Mapping

import polars as pl

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.tables import UnifiedTextDataTable


class CreateUnifiedTextDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        train_data: pl.DataFrame = context["train_data"]

        unified_text_data = train_data.with_columns(
            pl.concat_str(
                [
                    pl.lit("Question: \n"),
                    pl.col("prompt_title"),
                    pl.col("prompt_text"),
                    pl.col("prompt_question"),
                    pl.lit("Student Answer: \n"),
                    pl.col("text"),
                    pl.lit("---Grading---\n"),
                    pl.lit("Wording: "),
                    pl.col("wording"),
                    pl.lit("\nContent: "),
                    pl.col("content"),
                ]
            ).alias("unified_text")
        )

        table_io.write_table(unified_text_data, UnifiedTextDataTable())

        return {"unified_text_data": unified_text_data}
