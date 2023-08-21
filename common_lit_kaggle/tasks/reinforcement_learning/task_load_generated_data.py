import random
from typing import Any, Mapping

import polars as pl

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tables import RLGPT2SyntheticData

try:
    import nlpaug.augmenter.char as nchar
    import nlpaug.augmenter.word as naw
except ImportError:
    print("Could not import nlpaug")


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
            return 1.0 + (3 * random.random())

        # TODO: assign a degradation factor, df, where df = (4 - label)
        # Use it to create different between different randomized labels
        synthetic_dataframe = synthetic_dataframe.with_columns(
            pl.col("content").apply(randomize_label).alias("content"),
            pl.col("wording").apply(randomize_label).alias("wording"),
        )

        config = Config.get()

        augmenters = []
        for probabiltiies in [0.1, 0.2, 0.4]:
            augmenters.append(
                nchar.RandomCharAug(
                    action="substitute",
                    aug_char_p=probabiltiies,
                    aug_word_p=probabiltiies,
                    aug_word_max=30,
                    aug_word_min=probabiltiies * 10,
                )
            )

        def apply_degradation(row):
            text = row[2]
            synthetic_label = row[3]
            if synthetic_label <= 1.5:
                aug: nchar.RandomCharAug = augmenters[-1]
            elif synthetic_label <= 2.0:
                aug: nchar.RandomCharAug = augmenters[-2]
            elif synthetic_label <= 3.0:
                aug: nchar.RandomCharAug = augmenters[-3]
            else:
                # Unmodified data
                return row
            result = aug.augment(text, n=1)[0]
            as_list = list(row)
            as_list[2] = result
            return tuple(as_list)

        synthetic_dataframe = (
            synthetic_dataframe.apply(apply_degradation)
            .rename(
                {
                    "column_0": "student_id",
                    "column_1": "prompt_id",
                    "column_2": "text",
                    "column_3": "content",
                    "column_4": "wording",
                    "column_5": "prompt_question",
                    "column_6": "prompt_title",
                    "column_7": "prompt_text",
                    "column_8": "unified_text",
                    "column_9": "len_text",
                }
            )
            .with_columns(pl.col("len_text").cast(pl.UInt32))
        )

        print(train_data.columns)
        print(synthetic_dataframe.columns)
        print("Original train data", len(train_data))
        print("Synthetic data", len(synthetic_dataframe))
        augmented = pl.concat([train_data, synthetic_dataframe])
        print("Merged", len(augmented))
        return {"train_unified_text_data": augmented}
