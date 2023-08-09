import os

import torch
from torch import nn
from transformers import BartModel
from transformers.models.bart.configuration_bart import BartConfig

from common_lit_kaggle.modeling.base_regression import LinearHead
from common_lit_kaggle.settings.config import Config

# pylint: disable=no-member,too-many-ancestors
# pylint: disable=invalid-name,consider-using-f-string


class BartStackWithRegressionHead(nn.Module):
    """This code is a ripoff of the class BartClassificationHead from transformers library:

    https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py#L1474
    """

    def __init__(self, config: BartConfig, barts: list[BartModel]):
        super().__init__()

        project_config = Config.get()

        self.barts = barts

        self.regression_head = LinearHead(
            input_dim=config.d_model * len(self.barts),
            inner_dim=config.d_model * len(self.barts),
            num_classes=project_config.num_of_labels,
            dropout=project_config.dropout,
        )
        self._bart_config = config

    def save_pretrained(self, checkpoint_path):
        try:
            os.makedirs(checkpoint_path)
        except FileExistsError:
            pass

        for idx, bart in self.barts:
            bart.save_pretrained(checkpoint_path / f"_idx_{idx}")
        torch.save(self.regression_head, checkpoint_path / "regression_head.pt")

    def _forward_bart(self, bart, input_ids: torch.Tensor):
        outputs = bart(input_ids)
        print("output shape", outputs[0].shape)
        hidden_states = outputs[0]  # last hidden state

        # Black magic from Transformers library source code
        eos_mask = input_ids.eq(self._bart_config.eos_token_id).to(hidden_states.device)
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")

        print("mask shape", eos_mask.shape)
        sentence_representation = hidden_states[eos_mask, :].view(
            hidden_states.size(0), -1, hidden_states.size(-1)
        )[:, -1, :]
        print("sentence repr shape", sentence_representation.shape)
        return sentence_representation

    def forward(self, input_ids: torch.LongTensor):
        splits = torch.split(input_ids, self._bart_config.d_model, dim=1)
        print(" splits ", len(splits))
        sentences = []
        for idx, split in enumerate(splits):
            print("Split shape", split.shape)
            sentence = self._forward_bart(self.barts[idx], split)
            sentences.append(sentence)
        logits = self.regression_head(torch.cat(sentences))

        return logits
