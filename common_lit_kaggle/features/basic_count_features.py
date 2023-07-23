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

    def word_intersection_percentage(row):
        student_id = row[0]
        prompt_id = row[1]
        text = row[2]
        content = row[3]
        wording = row[4]
        prompt_question = row[5]
        prompt_title = row[6]
        prompt_text = row[7]

        text_words = set(text.split(" "))

        prompt_words = set(prompt_text.split(" "))

        # How many words of the prompt are repeated
        return len(text_words.intersection(prompt_words)) / len(prompt_words)

    word_intersection = input_data.apply(word_intersection_percentage)

    input_data = input_data.with_columns(
        word_intersection.select("apply").to_series().alias("word_intersection")
    )
    print(input_data)

    return input_data
