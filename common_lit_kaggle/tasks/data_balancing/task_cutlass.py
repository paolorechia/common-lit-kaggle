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


class CutlassTask(Task):
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

        target_statistic = 5.0
        max_iter = int(len(augmented_data) * 0.8)
        for _ in range(max_iter):
            # pylint: disable=singleton-comparison

            sampling = sampling.filter(pl.col("enabled") == True)
            num_data_points = len(sampling)
            print("Number of data points: ", num_data_points)
            content_test = compute_anderson_test(sampling, "content")
            print("Anderson test: ")
            print(content_test.statistic)

            if content_test.statistic < target_statistic:
                print("Target significance reached")
                break

            candidate_statistic = 100000
            for _ in range(max_iter):
                to_remove = random.randint(0, num_data_points)

                candidate = (
                    sampling.drop("enabled")
                    .with_row_count("row_nr")
                    .select(
                        "*",
                        pl.when(pl.col("row_nr") == to_remove)
                        .then(False)
                        .otherwise(pl.lit(True))
                        .alias("enabled"),
                    )
                    .drop("row_nr")
                    .filter(pl.col("enabled") == True)
                )
                assert len(candidate) < len(sampling)

                candidate_test = compute_anderson_test(candidate, "content")
                candidate_statistic = candidate_test.statistic
                if candidate_statistic < (content_test.statistic * 0.95):
                    # If we found an improvement, pick it
                    print("Found candidate!", candidate_statistic)
                    print("Number of data points of candidate: ", len(candidate))
                    sampling = candidate
                    break
        print(sampling)
        # Replace train data in pipeline with undersampled
        return {"train_data": sampling}
