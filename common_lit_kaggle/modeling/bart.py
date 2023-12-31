import torch
from torch import nn
from transformers import BartModel
from transformers.models.bart.configuration_bart import BartConfig

from common_lit_kaggle.modeling.base_regression import LinearHead
from common_lit_kaggle.settings.config import Config

# pylint: disable=no-member,too-many-ancestors
# pylint: disable=invalid-name,consider-using-f-string


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
            dropout=project_config.dropout,
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
