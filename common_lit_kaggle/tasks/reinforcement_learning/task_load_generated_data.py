import random
from typing import Any, Mapping

import polars as pl

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.tables import RLGPT2SyntheticData


class LoadGeneratedDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        train_data: pl.DataFrame = context["train_unified_text_data"]
        synthetic_dataframe: pl.DataFrame = table_io.read_table(RLGPT2SyntheticData())
        # Assume synthetic data has an above average score
        assumed_synthetic_score = 3.0

        synthetic_dataframe = (
            synthetic_dataframe.rename({"text": "text2"})
            .with_columns(
                pl.lit("SYNTHETIC").alias("student_id"),
                pl.lit("SYNTHETIC").alias("prompt_id"),
                pl.col("text2").alias("text"),
                pl.lit(assumed_synthetic_score).alias("content"),
                pl.lit(assumed_synthetic_score).alias("wording"),
                pl.lit("SYNTHETIC").alias("prompt_question"),
                pl.lit("SYNTHETIC").alias("prompt_title"),
                pl.lit("SYNTHETIC").alias("prompt_text"),
                pl.col("text2").alias("unified_text"),
                pl.col("text2").str.lengths().alias("len_text"),
            )
            .drop("text2")
        )

        random.seed(42)

        def randomize_label(_):
            return 2.0 + (2 * random.random())

        synthetic_dataframe = synthetic_dataframe.with_columns(
            pl.col("content").apply(randomize_label).alias("content"),
            pl.col("wording").apply(randomize_label).alias("wording"),
        )

        print(train_data.columns)
        print(synthetic_dataframe.columns)
        print("Original train data", len(train_data))
        print("Synthetic data", len(synthetic_dataframe))
        augmented = pl.concat([train_data, synthetic_dataframe])
        print("Merged", len(augmented))
        return {"train_unified_text_data": augmented}
