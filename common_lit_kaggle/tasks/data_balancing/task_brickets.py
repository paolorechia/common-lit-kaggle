import random
from typing import Any, Mapping

import numpy as np
import polars as pl
import scipy

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config


def compute_anderson_test(data: pl.DataFrame, label="content"):
    content_labels = data.select(label).to_numpy().reshape(-1)
    content_desired_uniform = np.array(
        [
            np.random.uniform(content_labels.min(), content_labels.max())
            for _ in range(len(content_labels))
        ]
    )

    content_test = scipy.stats.anderson_ksamp([content_labels, content_desired_uniform])
    return content_test


def sample_one(data: pl.DataFrame):
    to_take = random.randint(0, len(data))
    taken = (
        data.drop("enabled")
        .with_row_count("row_nr")
        .filter(pl.col("row_nr") == to_take)
    )
    return taken


def sample_n(data: pl.DataFrame, number_samples=10):
    sampled = []

    taken_idx = set()
    for _ in range(number_samples):
        found_unique = False

        for _ in range(1000):
            to_take = random.randint(0, len(data))
            if to_take not in taken_idx:
                found_unique = True
                taken_idx.add(to_take)
                break

        if not found_unique:
            raise ValueError("Could not sample unique value!")

        taken = (
            data.drop("enabled")
            .with_row_count("row_nr")
            .filter(pl.col("row_nr") == to_take)
        )
        sampled.append(taken)

    return list(taken_idx), pl.concat(sampled)


class BricketsTask(Task):
    def __init__(
        self, augmented_table: TableReference, name: str | None = None
    ) -> None:
        super().__init__(name)
        self.augmented_table = augmented_table

    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        config = Config.get()
        augmented_data = table_io.read_table(self.augmented_table)
        augmented_data = augmented_data.drop("text")
        augmented_data = augmented_data.rename({"augmented_text": "text"})

        sampling = augmented_data.with_columns(pl.lit(True).alias("enabled"))
        result = None

        target_statistic = 5.0
        max_iter = int(len(augmented_data) * 0.8)
        for _ in range(max_iter):
            # pylint: disable=singleton-comparison
            num_data_points = len(sampling)

            print("Remaining number of data points: ", num_data_points)

            if result is not None:
                content_test = compute_anderson_test(result, "content")
                print("Anderson test: ")
                print(content_test.statistic)

            number_to_sample = 100
            target_statistic = 1.0
            sampling_attempts = 1000

            for _ in range(sampling_attempts):
                idx_to_flag_as_taken, taken_sample = sample_n(
                    sampling, number_to_sample
                )

                if result is None:
                    taken_test = compute_anderson_test(taken_sample)
                    if taken_test.statistic < target_statistic:
                        result = taken_sample
                        print("Took result: ", taken_test)
                        break
                else:
                    candidate_result = pl.concat([result, taken_sample])
                    candidate_test = compute_anderson_test(candidate_result)
                    if candidate_test.statistic < target_statistic:
                        print("Sampled test: ", candidate_test)
                        result = candidate_result
                        break

            assert result
            print("Sampled datapoints: ", len(result))
            new_sampling = sampling
            for idx in idx_to_flag_as_taken:
                new_sampling = (
                    new_sampling.drop("enabled")
                    .with_row_count("row_nr")
                    .select(
                        "*",
                        pl.when(pl.col("row_nr") == idx)
                        .then(False)
                        .otherwise(pl.lit(True))
                        .alias("enabled"),
                    )
                    .drop("row_nr")
                    .filter(pl.col("enabled") == True)
                )
            assert len(new_sampling) < len(
                sampling
            ), f"{len(new_sampling)}, {len(sampling)}"
            sampling = new_sampling

        print(taken_sample)
        # Replace train data in pipeline with undersampled
        return {"train_data": taken_sample}
