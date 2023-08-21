import os

import torch
from torch import nn
from transformers import DebertaV2Model
from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler

from common_lit_kaggle.modeling.base_regression import LinearHead
from common_lit_kaggle.settings.config import Config

# pylint: disable=no-member,too-many-ancestors
# pylint: disable=invalid-name,consider-using-f-string


class DebertaTwinsWithRegressionHead(nn.Module):
    def __init__(
        self,
        config: DebertaV2Config,
        deberta_prompt: DebertaV2Model,
        deberta_answer: DebertaV2Model,
    ):
        super().__init__()

        project_config = Config.get()

        self.pooler = ContextPooler(config)

        self.deberta_prompt = deberta_prompt
        self.deberta_answer = deberta_answer

        self.regression_head = LinearHead(
            input_dim=self.pooler.output_dim * 2,
            inner_dim=config.hidden_size * 2,
            num_classes=project_config.num_of_labels,
            dropout=project_config.dropout,
        )

        self._deberta_config = config

    def save_pretrained(self, checkpoint_path):
        try:
            os.makedirs(checkpoint_path)
        except FileExistsError:
            pass

        self.deberta_prompt.save_pretrained(checkpoint_path / "_prompt")
        self.deberta_answer.save_pretrained(checkpoint_path / "_answer")
        torch.save(self.regression_head, checkpoint_path / "regression_head.pt")

    def _forward_deberta(
        self, deberta_model: DebertaV2Model, input_ids: torch.LongTensor
    ):
        """Reference implementation:
        https://github.com/huggingface/transformers/blob/080a97119c0dabfd0fb5c3e26a872ad2958e4f77/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L1261
        """

        outputs = deberta_model.forward(input_ids)
        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        return pooled_output

    def forward(
        self, prompt_input_ids: torch.LongTensor, answer_input_ids: torch.LongTensor
    ):
        prompt_representation = self._forward_deberta(
            self.deberta_prompt, prompt_input_ids
        )
        answer_representation = self._forward_deberta(
            self.deberta_answer, answer_input_ids
        )

        batch_size = answer_representation.shape[0]
        # Flatten both barts output
        flattened = torch.cat([prompt_representation, answer_representation]).reshape(
            batch_size, -1
        )

        logits = self.regression_head(flattened)
        return logits
