import logging
from typing import Any, Mapping

import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from transformers import AutoModel, AutoTokenizer

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config

logger = logging.getLogger(__name__)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # pylint: disable=no-member
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def sentence_embedding_factory(model, tokenizer):
    config = Config.get()

    def calculate_sentence_embeddings(text):
        sentences = text.split(".")
        # Tokenize sentences
        encoded_input = tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input.to(config.device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.to("cpu").detach().numpy()

    return calculate_sentence_embeddings


def find_minimum_distance(
    text_embeddings_column, prompt_embeddings_column, metric="euclidean"
):
    num_rows = text_embeddings_column.shape[0]

    logger.info("Computing minimum distance...")
    output = []
    for i in range(num_rows):
        text_embeddings_row = text_embeddings_column[i][0]
        prompt_embeddings_row = prompt_embeddings_column[i][0]

        minimum_distance = pairwise_distances(
            text_embeddings_row, prompt_embeddings_row, metric=metric
        ).min()
        output.append(minimum_distance)
    return output


class _AddSentenceEmbeddingToDataSubTask:
    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        config = Config.get()

        logger.info("Loading tokenizer: %s", config.sentence_transformer)
        tokenizer = AutoTokenizer.from_pretrained(config.sentence_transformer)

        logger.info("Loading model: %s", config.sentence_transformer)
        model = AutoModel.from_pretrained(config.sentence_transformer)
        model.to(config.device)

        calculate_sentence_embeddings = sentence_embedding_factory(model, tokenizer)
        logger.info("Computing prompt text sentence embeddings...")
        data = data.with_columns(
            pl.col("prompt_text")
            .apply(calculate_sentence_embeddings)
            .alias("prompt_set_embedding")
        )

        logger.info("Computing text sentence embeddings...")
        data = data.with_columns(
            pl.col("text")
            .apply(calculate_sentence_embeddings)
            .alias("text_set_embedding")
        )
        logger.info(data)

        min_dist = find_minimum_distance(
            data.select("text_set_embedding").to_numpy(),
            data.select("prompt_set_embedding").to_numpy(),
        )

        assert len(min_dist) == len(data)

        data = data.with_columns(pl.Series(name="embedding_distance", values=min_dist))

        logger.info("Done!")
        logger.info(data)
        return data


class AddSentenceEmbeddingToTrainTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        # Load model from HuggingFace Hub
        key = "enriched_train_data"
        train_data: pl.DataFrame = context[key]
        train_data = _AddSentenceEmbeddingToDataSubTask().calculate(train_data)

        return {key: train_data, "extra_features": ["embedding_distance"]}


class AddSentenceEmbeddingToTestTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        # Load model from HuggingFace Hub
        key = "enriched_test_data"
        test_data: pl.DataFrame = context[key]
        test_data = _AddSentenceEmbeddingToDataSubTask().calculate(test_data)

        return {key: test_data, "extra_features": ["embedding_distance"]}


class AddSentenceEmbeddingToPredictTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        # Load model from HuggingFace Hub
        key = "enriched_prediction_data"
        test_data: pl.DataFrame = context[key]
        test_data = _AddSentenceEmbeddingToDataSubTask().calculate(test_data)

        return {
            "enriched_prediction_data": test_data,
            "extra_features": ["embedding_distance"],
        }
