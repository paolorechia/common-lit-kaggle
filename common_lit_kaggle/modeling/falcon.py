import logging

import torch
from torch import nn
from transformers import FalconConfig, FalconModel

from common_lit_kaggle.modeling.base_regression import LinearHead
from common_lit_kaggle.settings.config import Config

logger = logging.getLogger(__name__)

# pylint: disable=no-member


class FalconLoraWithRegressionHead(nn.Module):
    def __init__(self, config: FalconConfig, falcon_model: FalconModel):
        super().__init__()

        project_config = Config.get()
        self.falcon_model = falcon_model
        self.config = config
        self.regression_head = LinearHead(
            input_dim=config.hidden_size,
            inner_dim=config.hidden_size,
            num_classes=project_config.num_of_labels,
            dropout=project_config.dropout,
        )
        self.regression_head.to(project_config.device)

    def save_pretrained(self, checkpoint_path):
        self.falcon_model.save_pretrained(checkpoint_path)
        torch.save(self.regression_head, checkpoint_path / "regression_head.pt")

    def forward(self, input_ids: torch.LongTensor, *args, **kwargs):
        """Copied/adapted from the transformers library:

        https://github.com/huggingface/transformers/blob/05cda5df3405e6a2ee4ecf8f7e1b2300ebda472e/src/transformers/models/falcon/modeling_falcon.py#L979
        """
        kwargs["use_cache"] = False

        outputs = self.falcon_model.forward(input_ids, *args, **kwargs)
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
