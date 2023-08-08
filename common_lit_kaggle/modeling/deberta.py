import torch
from torch import nn
from transformers import DebertaV2Model
from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler

from common_lit_kaggle.modeling.base_regression import LinearHead
from common_lit_kaggle.settings.config import Config

# pylint: disable=no-member,too-many-ancestors
# pylint: disable=invalid-name,consider-using-f-string


class DebertaWithRegressionHead(DebertaV2Model):
    def __init__(self, config: DebertaV2Config):
        super().__init__(config)
        project_config = Config.get()

        self.pooler = ContextPooler(config)
        self.regression_head = LinearHead(
            input_dim=self.pooler.output_dim,
            inner_dim=config.hidden_size,
            num_classes=project_config.num_of_labels,
            dropout=project_config.dropout,
        )

    def forward(self, input_ids: torch.LongTensor):
        """Reference implementation:
        https://github.com/huggingface/transformers/blob/080a97119c0dabfd0fb5c3e26a872ad2958e4f77/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L1261
        """

        outputs = super().forward(input_ids)
        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        logits = self.regression_head(pooled_output)

        return logits
