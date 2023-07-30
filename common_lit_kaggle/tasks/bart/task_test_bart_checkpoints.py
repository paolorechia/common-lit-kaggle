"""Train code adapted from PyTorch tutorial:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

"""
import logging
from typing import Any, Mapping, Optional

import polars as pl
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.modeling import BartWithRegressionHead
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.utils.checkpoint import get_checkpoint_path
from common_lit_kaggle.utils.mlflow_wrapper import mlflow

logger = logging.getLogger(__name__)

# pylint: disable=no-member,too-many-ancestors
# pylint: disable=invalid-name,consider-using-f-string


class TestBartCheckpointsTask(Task):
    def __init__(self, name: str | None = None, existing_run_id: str = "") -> None:
        super().__init__(name)
        self.truncation_length: Optional[int] = None
        self.existing_run_id = existing_run_id
        assert self.existing_run_id, "Must pass an existing run id to test"

    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        config = Config.get()

        with mlflow.start_run(run_id=self.existing_run_id) as _:
            tensors_to_predict = context["predict_input_ids_stack"]
            prediction_data: pl.DataFrame = context["test_data"]

            for epoch in range(1, config.num_train_epochs + 1):
                print("Testing epoch: ", epoch)

                checkpoint_path = get_checkpoint_path(
                    passed_model_name=config.bart_model,
                    existing_run_id=self.existing_run_id,
                    epoch=epoch,
                )

                bart_model = BartWithRegressionHead.from_pretrained(checkpoint_path)
                bart_model.to(config.device)

                bart_model.eval()

                content = []
                wording = []

                config = Config.get()
                batch_size = config.batch_size
                # TODO: make this prediction work in batches too
                for tensor in tqdm(tensors_to_predict):
                    tensor = tensor.to(config.device)

                    result = bart_model.forward(
                        tensor.reshape(1, config.model_context_length)
                    )
                    content.append(result.cpu().detach()[0][0])
                    wording.append(result.cpu().detach()[0][1])

                    tensor = tensor.to("cpu")

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
