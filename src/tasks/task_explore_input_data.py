from itertools import product
from typing import Any, List, Mapping

import matplotlib.pyplot as plt
import polars as pl

from features import add_basic_features
from framework import table_io
from framework.task import Task
from settings import config
from tables import StudentsPerTextTable, TextsPerPromptTable


class ExploreInputDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data: pl.DataFrame = context["joined_data"]

        print(input_data)

        # Finds out we have one text per student
        texts_per_student = input_data.groupby("student_id").agg(pl.count()).describe()
        table_io.write_table(texts_per_student, StudentsPerTextTable())

        # Checks how many prompts we have
        texts_per_prompt = input_data.groupby(
            "prompt_id", "prompt_question", "prompt_title"
        ).agg(pl.count())

        # Prompt
        table_io.write_table(texts_per_prompt, TextsPerPromptTable())

        # Add basic features to input_data
        input_data = add_basic_features(input_data)

        # Spit dataset per prompt
        prompts = texts_per_prompt.select("prompt_title").unique().to_series().to_list()

        labels = [
            "content",
            "wording",
        ]
        features = [
            "text_length",
            "word_count",
            "sentence_count",
            "unique_words",
        ]
        # Generate plots for each prompt
        for prompt in prompts:
            self.plots(input_data, prompt, labels, features)

        return {}

    def plots(
        self,
        input_data: pl.DataFrame,
        prompt: str,
        labels: List[str],
        features: List[str],
    ):
        normalized_prompt = prompt.replace(" ", "_").lower()
        print("prompt:", prompt)
        text_from_prompt = input_data.filter(pl.col("prompt_title") == prompt)
        print(text_from_prompt)
        attributes = labels + features

        self.generate_histogram(text_from_prompt, normalized_prompt, attributes)

        # Scatter plot of labels
        self.label_scatter(text_from_prompt, normalized_prompt)

        # Scatter label x feature
        label_feature_pairs = product(labels, features)
        for pair in label_feature_pairs:
            self.pair_scatter(text_from_prompt, normalized_prompt, pair)

    def generate_histogram(
        self,
        text_from_prompt: pl.DataFrame,
        normalized_prompt: str,
        attributes: List[str],
    ):
        # Generate histograms
        for attribute in attributes:
            attr_numpy = text_from_prompt.select(pl.col(attribute)).to_numpy()
            fig, axis = plt.subplots()
            axis.set_ylabel("frequency")
            axis.set_xlabel(attribute)
            axis.hist(attr_numpy, bins=50)
            plot_path = config.PLOTS_DIR / (
                f"{attribute}_distribution_" + normalized_prompt + ".jpg"
            )
            fig.savefig(plot_path)

    def label_scatter(self, text_from_prompt: pl.DataFrame, normalized_prompt: str):
        wording = text_from_prompt.select(pl.col("wording")).to_numpy()
        content = text_from_prompt.select(pl.col("content")).to_numpy()
        fig, axis = plt.subplots()
        axis.set_xlabel("wording")
        axis.set_ylabel("content")
        axis.scatter(wording, content)

        plot_path = config.PLOTS_DIR / ("labels_scatter_" + normalized_prompt + ".jpg")
        fig.savefig(plot_path)

    def pair_scatter(
        self,
        text_from_prompt: pl.DataFrame,
        normalized_prompt: str,
        pair: tuple[str, str],
    ):
        pair_x = text_from_prompt.select(pl.col(pair[0])).to_numpy()
        pair_y = text_from_prompt.select(pl.col(pair[1])).to_numpy()
        fig, axis = plt.subplots()
        axis.set_xlabel(pair[0])
        axis.set_ylabel(pair[1])
        axis.scatter(pair_x, pair_y)

        plot_path = config.PLOTS_DIR / (
            "pair_scatter_" + f"{pair[0]}_x_{pair[1]}_" + normalized_prompt + ".jpg"
        )
        fig.savefig(plot_path)
