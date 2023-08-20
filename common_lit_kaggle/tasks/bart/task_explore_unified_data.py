from typing import Any, Mapping

import matplotlib.pyplot as plt
import polars as pl

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config


class ExploreUnifiedInputDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data: pl.DataFrame = context["train_unified_text_data"]
        text_length = input_data.select(
            pl.col("unified_text").str.lengths().alias("input_text_length")
        ).sort(by=pl.col("input_text_length"))
        self.plot(text_length, "input_text_length")
        self.plot(input_data, "content", bins=8)
        self.plot(input_data, "wording", bins=8)
        return {}

    def plot(self, text_length: pl.DataFrame, attribute: str, bins=None):
        config = Config.get()

        # Generate histograms
        attr_numpy = text_length.select(pl.col(attribute)).to_numpy()
        fig, axis = plt.subplots()
        axis.set_ylabel("length")
        axis.set_xlabel(attribute)
        if not bins:
            bins = len(text_length)
        axis.hist(attr_numpy, bins=bins)
        plot_path = config.plots_dir / (f"length_{attribute}.jpg")
        fig.savefig(plot_path)
