import os

import torch
from torch import nn
from transformers import BartModel
from transformers.models.bart.configuration_bart import BartConfig

from common_lit_kaggle.modeling.base_regression import LinearHead
from common_lit_kaggle.settings.config import Config

# pylint: disable=no-member,too-many-ancestors
# pylint: disable=invalid-name,consider-using-f-string


class BartTwinsWithRegressionHead(nn.Module):
    def __init__(
        self, config: BartConfig, bart_prompt: BartModel, bart_answer: BartModel
    ):
        super().__init__()

        project_config = Config.get()

        self.bart_prompt = bart_prompt
        self.bart_answer = bart_answer

        self.regression_head = LinearHead(
            input_dim=config.d_model * 2,
            inner_dim=config.d_model * 2,
            num_classes=project_config.num_of_labels,
            dropout=project_config.dropout,
        )
        self._bart_config = config

    def save_pretrained(self, checkpoint_path):
        try:
            os.makedirs(checkpoint_path)
        except FileExistsError:
            pass

        self.bart_prompt.save_pretrained(checkpoint_path / "_prompt")
        self.bart_answer.save_pretrained(checkpoint_path / "_answer")
        torch.save(self.regression_head, checkpoint_path / "regression_head.pt")

    def _forward_bart(self, bart: BartModel, input_ids: torch.LongTensor):
        outputs = bart.forward(input_ids)
        hidden_states = outputs[0]  # last hidden state

        # Black magic from Transformers library source code
        eos_mask = input_ids.eq(bart.config.eos_token_id).to(hidden_states.device)
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")

        sentence_representation = hidden_states[eos_mask, :].view(
            hidden_states.size(0), -1, hidden_states.size(-1)
        )[:, -1, :]

        return sentence_representation

    def forward(
        self, prompt_input_ids: torch.LongTensor, answer_input_ids: torch.LongTensor
    ):
        prompt_representation = self._forward_bart(self.bart_prompt, prompt_input_ids)
        answer_representation = self._forward_bart(self.bart_answer, answer_input_ids)

        batch_size = answer_representation.shape[0]
        # Flatten both barts output
        flattened = torch.cat([prompt_representation, answer_representation]).reshape(
            batch_size, -1
        )
        logits = self.regression_head(flattened)
        return logits
