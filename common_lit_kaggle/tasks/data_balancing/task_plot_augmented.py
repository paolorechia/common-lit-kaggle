from typing import Any, List, Mapping

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tables import TrainSplitTable

class TaskPlotAugmented(Task):
    def __init__(
        self, augmented_table: TableReference, name: str | None = None
    ) -> None:
        super().__init__(name)
        self.augmented_table = augmented_table

    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = table_io.read_table(TrainSplitTable())
        augmented_data = table_io.read_table(self.augmented_table)
        augmented_data = augmented_data.drop("text")
        augmented_data = augmented_data.rename({"augmented_text": "text"})

        concat_data = pl.concat(
            [
                input_data.select(sorted(input_data.columns)),
                augmented_data.select(sorted(augmented_data.columns)),
            ]
        )
        self.generate_histogram_of_label(
            concat_data, self.augmented_table.name, ["content", "wording"]
        )

        content_labels = augmented_data.select("content").to_numpy().reshape(-1)
        wording_labels = augmented_data.select("wording").to_numpy().reshape(-1)
        print("content shape", content_labels.shape)

        content_desired_uniform = np.array(
            [
                np.random.uniform(content_labels.min(), content_labels.max())
                for _ in range(len(content_labels))
            ]
        )

        print("desired shape", content_desired_uniform.shape)
        print("Content check")
        content_test = scipy.stats.anderson_ksamp(
            [content_labels, content_desired_uniform]
        )
        print(content_test.statistic, content_test.significance_level)
        print(content_test.critical_values)

        wording_desired_uniform = np.array(
            [
                np.random.uniform(wording_labels.min(), wording_labels.max())
                for _ in range(len(wording_labels))
            ]
        )

        print("Wording check")
        wording_test = scipy.stats.anderson_ksamp(
            [wording_labels, wording_desired_uniform]
        )
        print(wording_test.statistic, wording_test.significance_level)
        print(wording_test.critical_values)

        return {}

    def generate_histogram_of_label(
        self,
        input_data: pl.DataFrame,
        table_name: str,
        attributes: List[str],
    ):
        config = Config.get()

        for attribute in attributes:
            attr_numpy = input_data.select(pl.col(attribute)).to_numpy()
            fig, axis = plt.subplots()
            axis.set_ylabel("frequency")
            axis.set_xlabel(attribute)
            axis.hist(attr_numpy, bins=8)
            plot_path = config.plots_dir / (
                f"augmented_{table_name}_{attribute}_histogram.jpg"
            )
            fig.savefig(plot_path)
