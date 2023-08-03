from typing import Any, List, Mapping

import matplotlib.pyplot as plt
import polars as pl

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config


class TaskPlotAugmented(Task):
    def __init__(
        self, augmented_table: TableReference, name: str | None = None
    ) -> None:
        super().__init__(name)
        self.augmented_table = augmented_table

    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        table = table_io.read_table(self.augmented_table)
        self.generate_histogram_of_label(
            table, self.augmented_table.name, ["content", "wording"]
        )
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
