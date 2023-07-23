from typing import Any, Mapping

import matplotlib.pyplot as plt
import polars as pl

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

        prompts = texts_per_prompt.select("prompt_title").unique().to_series().to_list()

        for prompt in prompts:
            print("prompt:", prompt)
            text_from_prompt = input_data.filter(pl.col("prompt_title") == prompt)
            for attribute in ["content", "wording"]:
                content = text_from_prompt.select(pl.col(attribute)).to_numpy()
                fig, axis = plt.subplots()
                axis.set_ylabel("frequency")
                axis.set_xlabel(attribute)
                axis.hist(content)
                normalized_prompt = prompt.replace(" ", "_").lower()
                plot_path = config.PLOTS_DIR / (
                    f"{attribute}_distribution_" + normalized_prompt + ".jpg"
                )
                fig.savefig(plot_path)

        return {}
