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

        for idx, bart in enumerate(self.barts):
            bart.save_pretrained(checkpoint_path / f"_idx_{idx}")
        torch.save(self.regression_head, checkpoint_path / "regression_head.pt")

    def _forward_bart(self, bart, input_ids: torch.Tensor):
        outputs = bart(input_ids)
        hidden_states = outputs[0]  # last hidden state

        # TODO: need to fix this block for the stack
        # Problem: eos tokens only exist for one of the bart models
        #
        # How do we create a mask when there is no eos token
        # Do create an arbitrary mask?
        # Do we just take the last row of the hidden_states?

        # TODO: check if taking just last hidden state row works
        # pylint: disable=broad-exception-caught
        try:
            # Black magic from Transformers library source code
            eos_mask = input_ids.eq(self._bart_config.eos_token_id).to(
                hidden_states.device
            )
            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError(
                    "All examples must have the same number of <eos> tokens."
                )

            sentence_representation = hidden_states[eos_mask, :].view(
                hidden_states.size(0), -1, hidden_states.size(-1)
            )[:, -1, :]
        except Exception:
            # No valid eos mask, just take last state
            sentence_representation = hidden_states[:, -1, :]
        return sentence_representation

    def forward(self, input_ids: torch.LongTensor):
        splits = torch.split(input_ids, self._bart_config.d_model, dim=1)
        sentences = []
        for idx, split in enumerate(splits):
            sentence = self._forward_bart(self.barts[idx], split)
            sentences.append(sentence)
        logits = self.regression_head(torch.cat(sentences, dim=1))

        return logits
