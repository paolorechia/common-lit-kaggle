"""Train code adapted from PyTorch tutorial:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

"""
import logging
from typing import Any, Mapping, Optional

import mlflow
import polars as pl
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.modeling import BartWithRegressionHead
from common_lit_kaggle.settings.config import Config

logger = logging.getLogger(__name__)

# pylint: disable=no-member,too-many-ancestors
# pylint: disable=invalid-name,consider-using-f-string


class TestBartCheckpointsTask(Task):
    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.truncation_length: Optional[int] = None

    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        config = Config.get()

        tensors_to_predict = context["predict_input_ids_stack"]
        prediction_data: pl.DataFrame = context["test_data"]

        for epoch in range(1, config.num_train_epochs + 1):
            print("Testing epoch: ", epoch)
            bart_path = f"trained_bart_{epoch}"

            bart_model = BartWithRegressionHead.from_pretrained(bart_path)

            # Use limit to test it quickly
            tensors_to_predict = tensors_to_predict[:100]
            prediction_data = prediction_data.limit(n=100)
            bart_model.eval()

            content = []
            wording = []

            config = Config.get()
            batch_size = config.batch_size
            # TODO: make this prediction work in batches too
            for tensor in tqdm(tensors_to_predict):
                result = bart_model.forward(tensor.reshape(1, 768))
                content.append(result.cpu().detach()[0][0])
                wording.append(result.cpu().detach()[0][1])

            logger.info("Starting prediction")

            data_with_predictions = prediction_data.with_columns(
                pl.Series("content_preds", content)
            )
            data_with_predictions = data_with_predictions.with_columns(
                pl.Series("wording_preds", wording)
            )

            wording_score = mean_squared_error(
                data_with_predictions.select("wording_preds").to_numpy(),
                data_with_predictions.select("wording").to_numpy(),
                squared=True,
            )

            logger.info("Wording error: %s", wording_score)
            mlflow.log_metric("wording_mean_squared_error", wording_score, epoch)

            content_score = mean_squared_error(
                data_with_predictions.select("content_preds").to_numpy(),
                data_with_predictions.select("content").to_numpy(),
                squared=True,
            )

            logger.info("Content error: %s", content_score)
            mlflow.log_metric("content_mean_squared_error", content_score, epoch)

            mean = (content_score + wording_score) / 2
            logger.info("Mean error: %s", mean)
            mlflow.log_metric("avg_mean_squared_error", mean, epoch)

        return {"data_with_predictions": prediction_data}
