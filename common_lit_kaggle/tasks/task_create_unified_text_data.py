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
                    pl.lit("TOPIC TITLE: "),
                    pl.col("prompt_title"),
                    pl.lit("\nREFERENCE TEXT: "),
                    pl.col("prompt_text"),
                    pl.lit("\nQUESTION: \n"),
                    pl.col("prompt_question"),
                    pl.lit("\nSTUDENT ANSWER: \n"),
                    pl.col("text"),
                    pl.lit("\n\nGRADING SECTION"),
                    pl.lit("\nWORDING: "),
                    pl.col("wording"),
                    pl.lit("\nCONTENT: "),
                    pl.col("content"),
                    pl.lit("\nEND_OF_TEXT\n"),
                ]
            ).alias("unified_text")
        )

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

        return {"unified_text_data": unified_text_data}
