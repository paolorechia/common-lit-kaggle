"""Train code adapted from PyTorch tutorial:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

"""
import logging
from typing import Any, Mapping, Optional

import polars as pl
from tqdm import tqdm

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.modeling import BartWithRegressionHead
from common_lit_kaggle.settings.config import Config

logger = logging.getLogger(__name__)


class PredictBertTask(Task):
    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.truncation_length: Optional[int] = None

    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        config = Config.get()

        bart_path = "trained_bart"
        tensors_to_predict = context["predict_input_ids_stack"]
        prediction_data: pl.DataFrame = context["test_data"]

        if config.run_with_small_sample:
            prediction_data = prediction_data.limit(config.small_sample_size)

        bart_model = BartWithRegressionHead.from_pretrained(bart_path)
        bart_model.eval()

        content = []
        wording = []

        batch_size = config.batch_size
        # TODO: make this prediction work in batches too
        for tensor in tqdm(tensors_to_predict):
            result = bart_model.forward(tensor.reshape(1, 768))
            content.append(result.cpu().detach()[0][0])
            wording.append(result.cpu().detach()[0][1])

        logger.info("Starting prediction")

        prediction_data = prediction_data.with_columns(
            pl.Series("content_preds", content)
        )
        prediction_data = prediction_data.with_columns(
            pl.Series("wording_preds", wording)
        )

        return {"data_with_predictions": prediction_data}
