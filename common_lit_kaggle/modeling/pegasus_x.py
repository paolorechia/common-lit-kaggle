import torch
from torch import nn
from transformers import PegasusXConfig
from transformers.modeling_outputs import Seq2SeqModelOutput
from transformers.models.pegasus_x import PegasusXModel

from common_lit_kaggle.modeling.base_regression import LinearHead
from common_lit_kaggle.settings.config import Config

# pylint: disable=no-member,too-many-ancestors,arguments-differ,abstract-method
# pylint: disable=invalid-name,consider-using-f-string


class PegasusXWithRegressionHead(PegasusXModel):
    def __init__(self, config: PegasusXConfig):
        super().__init__(config)
        project_config = Config.get()
        self.regression_head = LinearHead(
            input_dim=config.d_model,
            inner_dim=config.d_model,
            num_classes=project_config.num_of_labels,
            dropout=project_config.dropout,
        )
        self._pegasus_config = config

    def forward(self, input_ids: torch.LongTensor):
        outputs: Seq2SeqModelOutput = super().forward(
            input_ids=input_ids, decoder_input_ids=input_ids
        )
        hidden_states = outputs.last_hidden_state

        # Black magic from Transformers library source code
        eos_mask = input_ids.eq(self._pegasus_config.eos_token_id).to(
            hidden_states.device
        )
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")

        sentence_representation = hidden_states[eos_mask, :].view(
            hidden_states.size(0), -1, hidden_states.size(-1)
        )[:, -1, :]

        logits = self.regression_head(sentence_representation)

        return logits
