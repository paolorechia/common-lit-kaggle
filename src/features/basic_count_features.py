from typing import List

import polars as pl


def add_basic_features(input_data) -> pl.DataFrame:
    input_data = input_data.with_columns(
        pl.col("text").str.n_chars().alias("text_length")
    )

    input_data = input_data.with_columns(
        pl.col("text").str.split(" ").list.lengths().alias("word_count")
    )

    input_data = input_data.with_columns(
        pl.col("text").str.split(".").list.lengths().alias("sentence_count")
    )

    def count_unique_words(words: List[str]):
        return len(set(words))

    input_data = input_data.with_columns(
        pl.col("text").str.split(" ").apply(count_unique_words).alias("unique_words")
    )
    return input_data
