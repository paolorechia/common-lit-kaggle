import logging

import torch
from torch import nn
from transformers import FalconModel
from transformers.models.falcon.configuration_falcon import FalconConfig

from common_lit_kaggle.settings.config import Config

# pylint: disable=no-member,too-many-ancestors
# pylint: disable=invalid-name,consider-using-f-string

logger = logging.getLogger(__name__)


class LinearHead(nn.Module):
    """Head for text regression task. This code comes from transformers source code."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()

        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class FalconWithRegressionHead(FalconModel):
    def __init__(self, config: FalconConfig):
        super().__init__(config)
        project_config = Config.get()

        self.regression_head = LinearHead(
            input_dim=config.d_model,
            inner_dim=config.d_model,
            num_classes=project_config.num_of_labels,
            pooler_dropout=config.classifier_dropout,
        )

    def forward(self, input_ids: torch.LongTensor):
        """Copied/adapted from the transformers library:

        https://github.com/huggingface/transformers/blob/05cda5df3405e6a2ee4ecf8f7e1b2300ebda472e/src/transformers/models/falcon/modeling_falcon.py#L979
        """
        outputs = super().forward(input_ids)
        hidden_states = outputs[0]  # last hidden state

        logits = self.regression_head.forward(hidden_states)

        batch_size = input_ids.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.ne(input_ids, self.config.pad_token_id).sum(dim=-1) - 1  # type: ignore
                ).to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    "%s will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`",
                    self.__class__.__name__,
                )

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        return pooled_logits
