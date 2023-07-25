from typing import Any, Mapping, Optional

import numpy as np
import polars as pl
import torch

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config


class PrepareTensorDataTask(Task):
    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.truncation_length: Optional[int] = None

    def set_string_length_truncation(self, truncation_length: int):
        self.truncation_length = truncation_length

    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        assert self.truncation_length, "Set string length truncation first!"
        config = Config.get()

        input_data: pl.DataFrame = context["unified_text_data"]

        # WIP: figure out how to load this data into pytorch with the right format

        # Should probably use the model tokenizer to create unified_text_data batches
        # But how to handle the labels?
        content = input_data.select("content").to_numpy().reshape(-1)
        wording = input_data.select("wording").to_numpy().reshape(-1)

        print(content)
        print(wording.shape)

        # creating tensor from targets_df
        content_tensor = torch.Tensor(content.astype(np.float64))
        wording_tensor = torch.Tensor(wording.astype(np.float64))

        print(content_tensor)
        # printing out result
        print(content_tensor.shape)
        print(wording_tensor.shape)

        return {}
