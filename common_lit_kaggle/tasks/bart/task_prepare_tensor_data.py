import logging
import warnings
from typing import Any, Mapping, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config

logger = logging.getLogger(__name__)


def tokenize_text(
    bart_tokenizer: AutoTokenizer,
    text: str,
    truncation_length: int,
    model_context_length: int,
):
    assert truncation_length, "Set string length truncation first!"

    assert isinstance(text, str)

    # pylint: disable=invalid-unary-operand-type
    used_text = text[-truncation_length:]

    assert len(used_text) <= truncation_length

    input_ids = bart_tokenizer(used_text, return_tensors="pt")["input_ids"]
    if len(input_ids) >= model_context_length:
        # If we ignore this error, we'll face a crash during training
        raise TypeError(
            f"Context model length limit not respected! {len(input_ids)} > {model_context_length}"
        )

    return input_ids


def dataframe_to_tensors(
    input_data: pl.DataFrame,
    bart_tokenizer: AutoTokenizer,
    truncation_length: int,
    model_context_length: int,
):
    input_ids_list = []
    for text in input_data.select(pl.col("unified_text")).to_numpy():
        input_ids: torch.Tensor = tokenize_text(
            bart_tokenizer,
            text[0],
            truncation_length,
            model_context_length,
        )

        padding_length = model_context_length - len(input_ids[0])
        if padding_length < 0:
            # If we let data exceet context size, we crash during training loop
            raise ValueError("Data does not fit into context size")

        padder = torch.nn.ConstantPad1d((0, padding_length), 0)
        padded = padder((input_ids)).reshape(-1)
        padded[0] = float(bart_tokenizer.eos_token_id)

        input_ids_list.append(padded)

    # pylint: disable=no-member
    input_ids_stack = torch.stack(input_ids_list)
    return input_ids_stack


class PrepareTensorTrainDataTask(Task):
    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.truncation_length: Optional[int] = None

    def set_string_length_truncation(self, truncation_length: int):
        self.truncation_length = truncation_length

    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        assert self.truncation_length, "Set string length truncation first!"

        config = Config.get()

        model_path = config.bart_model
        bart_tokenizer = AutoTokenizer.from_pretrained(model_path)

        input_data: pl.DataFrame = context["train_unified_text_data"]

        if config.run_with_small_sample:
            input_data = input_data.sample(config.small_sample_size)

        input_ids_stack = dataframe_to_tensors(
            input_data=input_data,
            bart_tokenizer=bart_tokenizer,
            truncation_length=config.string_truncation_length,
            model_context_length=config.model_context_length,
        )

        content = input_data.select("content").to_numpy().reshape(-1)
        wording = input_data.select("wording").to_numpy().reshape(-1)

        content_tensor = torch.Tensor(content.astype(np.float64))
        wording_tensor = torch.Tensor(wording.astype(np.float64))
        # pylint: disable=no-member

        stacked_labels = torch.stack(
            [
                content_tensor,
                wording_tensor,
            ],
            dim=1,
        )

        tensor_train_data = TensorDataset(
            input_ids_stack.to(config.device), stacked_labels.to(config.device)
        )
        # Managed to create a TensorDataset, but is this correct? :D
        return {
            "tokenizer": bart_tokenizer,
            "tensor_train_data": tensor_train_data,
        }


class PrepareTensorPredictDataTask(Task):
    def __init__(
        self,
        name: str = "",
        input_data_key: str = "test_unified_text_data",
        truncation_length: Optional[int] = None,
    ) -> None:
        super().__init__(name)
        self.truncation_length: Optional[int] = truncation_length
        self.input_data_key = input_data_key

    def set_string_length_truncation(self, truncation_length: int):
        self.truncation_length = truncation_length
        warnings.warn("Deprecated method, prefer passing this in the constructor")

    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        assert self.truncation_length, "Set string length truncation first!"

        config = Config.get()

        tokenizer_path = config.tokenizer
        bart_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        input_data: pl.DataFrame = context[self.input_data_key]
        if config.run_with_small_sample:
            input_data = input_data.limit(config.small_sample_size)

        input_ids_stack = dataframe_to_tensors(
            input_data=input_data,
            bart_tokenizer=bart_tokenizer,
            truncation_length=config.string_truncation_length,
            model_context_length=config.model_context_length,
        )

        return {"predict_input_ids_stack": input_ids_stack}
