import logging
from typing import Any, Mapping

import matplotlib.pyplot as plt

try:
    import mlflow
except ImportError:
    pass

import polars as pl
from sklearn.metrics import mean_squared_error

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config

logger = logging.getLogger(__name__)


def plot_labels_x_predictions(name, labels, predictions):
    config = Config.get()
    fig, axis = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    axis.set_ylabel("error")
    errors = labels[0:100] - predictions[0:100]

    axis.plot(errors)

    plot_path = config.plots_dir / f"{name}_labels_x_predictions.jpg"
    fig.savefig(plot_path)


class AnalysePredictionsTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        config = Config.get()

        logger.info("Analysing predictions...")

        data_with_predictions: pl.DataFrame = context["data_with_predictions"]
        mlflow.set_tag("name", self.name)
        for idx, feature in enumerate(config.used_features):
            mlflow.log_param(f"features_{idx}", feature)

        for idx, prompt in enumerate(config.train_prompts):
            mlflow.log_param(f"train_prompt_{idx}", prompt)

        mlflow.log_param("distance_metric", config.distance_metric)
        mlflow.log_param("sentence_transformer", config.sentence_transformer)
        mlflow.log_param("distance_stategy", config.distance_stategy)

        wording_score = mean_squared_error(
            data_with_predictions.select("wording_preds").to_numpy(),
            data_with_predictions.select("wording").to_numpy(),
            squared=True,
        )

        logger.info("Wording error: %s", wording_score)
        mlflow.log_metric("wording_mean_squared_error", wording_score)

        content_score = mean_squared_error(
            data_with_predictions.select("content_preds").to_numpy(),
            data_with_predictions.select("content").to_numpy(),
            squared=True,
        )

        logger.info("Content error: %s", content_score)
        mlflow.log_metric("content_mean_squared_error", content_score)

        mean = (content_score + wording_score) / 2
        logger.info("Mean error: %s", mean)
        mlflow.log_metric("avg_mean_squared_error", mean)

        logger.info("Plotting predictions...")
        plot_labels_x_predictions(
            "wording",
            data_with_predictions.select("wording_preds").to_numpy(),
            data_with_predictions.select("wording").to_numpy(),
        )

        plot_labels_x_predictions(
            "content",
            data_with_predictions.select("content_preds").to_numpy(),
            data_with_predictions.select("content").to_numpy(),
        )

        return {}
