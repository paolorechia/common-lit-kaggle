import torch
from torch import nn
from transformers import BartModel
from transformers.models.bart.configuration_bart import BartConfig

from common_lit_kaggle.settings.config import Config

# pylint: disable=no-member,too-many-ancestors
# pylint: disable=invalid-name,consider-using-f-string


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


class BartWithRegressionHead(BartModel):
    """This code is a ripoff of the class BartClassificationHead from transformers library:

    https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py#L1474
    """

    def __init__(self, config: BartConfig):
        super().__init__(config)
        project_config = Config.get()

        self.regression_head = LinearHead(
            input_dim=config.d_model,
            inner_dim=config.d_model,
            num_classes=project_config.num_of_labels,
            pooler_dropout=config.classifier_dropout,
        )

    def forward(self, input_ids: torch.LongTensor):
        outputs = super().forward(input_ids)
        hidden_states = outputs[0]  # last hidden state

        # Black magic from Transformers library source code
        eos_mask = input_ids.eq(self.config.eos_token_id).to(hidden_states.device)
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")

        sentence_representation = hidden_states[eos_mask, :].view(
            hidden_states.size(0), -1, hidden_states.size(-1)
        )[:, -1, :]
        logits = self.regression_head(sentence_representation)

        return logits
