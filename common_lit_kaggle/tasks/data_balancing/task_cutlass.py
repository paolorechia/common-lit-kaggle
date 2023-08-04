from typing import Any, Mapping

import numpy as np
import polars as pl
import random
import scipy

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config


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


        print("desired shape", content_desired_uniform.shape)
        print("Content check")

        sampling = augmented_data.with_columns(
            pl.lit(True).alias("enabled")
        )

        target_significance = 5.0 
        max_iter = 10
        for _ in range(max_iter):
            sampling = sampling.filter(
                pl.col("enabled") == True
            )
            num_data_points = len(sampling)
            print("Number of data points: ", num_data_points)
            content_labels = sampling.select("content").to_numpy().reshape(-1)
            content_desired_uniform = np.array(
                [
                    np.random.uniform(content_labels.min(), content_labels.max())
                    for _ in range(len(content_labels))
                ]
            )


            content_test = scipy.stats.anderson_ksamp(
                [content_labels, content_desired_uniform]
            )
            print("Anderson test: ")
            print(content_test.statistic, content_test.significance_level)
            print(content_test.critical_values)
            if content_test.significance_level < target_significance:
                print("Target significance reached")
                break

            mean = content_labels.mean()
            max = content_labels.max()
            min = content_labels.min()

            dist_to_left = mean - min
            dist_to_right = max - mean

                
            print("Dataset mean: ", mean)
            print("Dataset dist to left: ", mean)
            print("Dataset dist to right: ", mean)
            if dist_to_left > dist_to_right:
                right = num_data_points // 2
                left = 0
            else:
                right = num_data_points
                left = num_data_points // 2
            to_remove = random.randint(left, right)

            sampling = sampling.to_frame().with_row_count("row_nr").select(
                "*",
                pl.when(pl.col("row_nr") == to_remove)
                .then(False)
                .otherwise(pl.lit(True))
                .alias("enabled")
            )

        print(sampling)
        # Replace train data in pipeline with undersampled
        return {"train_data": sampling}
