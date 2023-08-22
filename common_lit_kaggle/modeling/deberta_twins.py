import os
import pathlib

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
        use_distance=False,
        freeze_prompt=True,
        freeze_answer=True,
        freeze_poolers=False,
    ):
        super().__init__()

        project_config = Config.get()

        self.deberta_prompt = deberta_prompt
        self.deberta_answer = deberta_answer
        self._deberta_config = config
        self.use_distance = use_distance

        self.pooler_prompt = ContextPooler(config)
        self.pooler_answer = ContextPooler(config)

        # Freezes layers
        to_freeze = []
        if freeze_prompt:
            to_freeze.append(self.deberta_prompt)
            if freeze_poolers:
                to_freeze.append(self.pooler_prompt)

        if freeze_answer:
            to_freeze.append(self.deberta_answer)
            if freeze_poolers:
                to_freeze.append(self.pooler_answer)

        for frozen in to_freeze:
            for param in frozen.parameters():
                param.requires_grad = False

        if use_distance:
            self.hidden_size = 2
            self.regression_head = LinearHead(
                input_dim=1,
                inner_dim=self.hidden_size,
                num_classes=project_config.num_of_labels,
                dropout=project_config.dropout,
            )
        else:
            self.hidden_size = config.hidden_size * 2
            self.regression_head = LinearHead(
                input_dim=self.pooler_answer.output_dim * 2,
                inner_dim=self.hidden_size,
                num_classes=project_config.num_of_labels,
                dropout=project_config.dropout,
            )

    def save_pretrained(self, checkpoint_path):
        try:
            os.makedirs(checkpoint_path)
        except FileExistsError:
            pass

        path = pathlib.Path(checkpoint_path)
        self.deberta_prompt.save_pretrained(path / "prompt")
        self.deberta_answer.save_pretrained(path / "answer")
        torch.save(self.pooler_prompt, path / "pooler_prompt.pt")
        torch.save(self.pooler_answer, path / "pooler_answer.pt")
        torch.save(self.regression_head, path / "regression_head.pt")

    def _forward_deberta(
        self, deberta_model: DebertaV2Model, pooler, input_ids: torch.LongTensor
    ):
        """Reference implementation:
        https://github.com/huggingface/transformers/blob/080a97119c0dabfd0fb5c3e26a872ad2958e4f77/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L1261
        """

        outputs = deberta_model.forward(input_ids)
        encoder_layer = outputs[0]
        pooled_output = pooler(encoder_layer)
        return pooled_output

    def forward(
        self, prompt_input_ids: torch.LongTensor, answer_input_ids: torch.LongTensor
    ):
        prompt_representation = self._forward_deberta(
            self.deberta_prompt, self.pooler_prompt, prompt_input_ids
        )
        answer_representation = self._forward_deberta(
            self.deberta_answer, self.pooler_answer, answer_input_ids
        )

        batch_size = answer_representation.shape[0]
        # Flatten both barts output
        flattened = torch.cat([prompt_representation, answer_representation]).reshape(
            batch_size, -1
        )
        # Distance implementation:
        if self.use_distance:
            dist = torch.cdist(prompt_representation, answer_representation)
            logits = self.regression_head(dist)
        else:
            logits = self.regression_head(flattened)
        return logits
