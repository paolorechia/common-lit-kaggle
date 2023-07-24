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

    input_data = input_data.with_columns(
        pl.col("prompt").str.n_chars().alias("prompt_length")
    )

    input_data = input_data.with_columns(
        pl.col("prompt").str.split(" ").list.lengths().alias("prompt_word_count")
    )

    input_data = input_data.with_columns(
        pl.col("prompt").str.split(".").list.lengths().alias("prompt_sentence_count")
    )

    input_data = input_data.with_columns(
        pl.col("prompt")
        .str.split(" ")
        .apply(count_unique_words)
        .alias("prompt_unique_words")
    )

    def word_intersection_percentage(row):
        """Using numeric indices here was a bad idea,
        because the prediction data has a different column layout (without labels).
        """
        text = row[2]

        text_words = set(text.split(" "))

        try:
            prompt_words = set(row[7].split(" "))
        except AttributeError:
            # If we're dealing with prediction label, adjust the index
            prompt_words = set(row[5].split(" "))

        # How many words of the prompt are repeated
        return len(text_words.intersection(prompt_words)) / len(prompt_words)

    word_intersection = input_data.apply(word_intersection_percentage)

    input_data = input_data.with_columns(
        word_intersection.select("apply").to_series().alias("word_intersection")
    )
    print(input_data)

    return input_data
