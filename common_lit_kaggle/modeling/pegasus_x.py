import torch
from transformers import PegasusXConfig
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
            input_dim=config.hidden_size,
            inner_dim=config.max_position_embeddings,
            num_classes=project_config.num_of_labels,
            dropout=project_config.dropout,
        )

    def forward(self, input_ids: torch.LongTensor):
        outputs = super().forward(input_ids=input_ids, decoder_input_ids=input_ids)
        hidden_states = outputs[0]
        logits = self.regression_head(hidden_states)
        return logits
