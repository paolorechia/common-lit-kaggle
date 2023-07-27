"""Train code adapted from PyTorch tutorial:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

"""
import logging
import math
import time
from typing import Any, Mapping, Optional

import polars as pl
import torch
from torch import nn, optim
from tqdm import tqdm
from transformers import BartModel
from transformers.models.bart.configuration_bart import BartConfig

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config

logger = logging.getLogger(__name__)

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


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))


def train_epoch(dataloader, model: BartWithRegressionHead, optimizer, criterion):
    """Adapted from: https://huggingface.co/docs/transformers/v4.26.1/training#training-loop"""
    total_loss = 0
    for data in tqdm(dataloader):
        input_tensor, target_tensor = data

        optimizer.zero_grad()
        logits = model.forward(input_tensor)

        # Compute loss here
        loss = criterion(logits, target_tensor)
        loss.backward()

        optimizer.step()

        # lr_scheduler.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(
    train_dataloader,
    model,
    n_epochs,
    learning_rate=0.001,
    print_every=1,
):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        print("Starting epoch: ", epoch)
        loss = train_epoch(train_dataloader, model, optimizer, criterion)
        print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(
                "%s (%d %d%%) %.4f"
                % (
                    timeSince(start, epoch / n_epochs),
                    epoch,
                    epoch / n_epochs * 100,
                    print_loss_avg,
                )
            )


class PredictBertTask(Task):
    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.truncation_length: Optional[int] = None

    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        bart_path = context["trained_bart_path"]
        tensors_to_predict = context["predict_input_ids_stack"]
        prediction_data = context["test_data"]
        bart_model = BartWithRegressionHead.from_pretrained(bart_path)

        bart_model.eval()

        content = []
        wording = []

        config = Config.get()
        batch_size = config.batch_size
        # TODO: make this prediction work in batches too
        for tensor in tqdm(tensors_to_predict):
            result = bart_model.forward(tensor.reshape(1, 768))
            content.append(result.cpu().detach()[0][0])
            wording.append(result.cpu().detach()[0][1])

        logger.info("Starting prediction")

        prediction_data = prediction_data.with_columns(pl.Series("content", content))
        prediction_data = prediction_data.with_columns(pl.Series("wording", wording))

        # For now, save some duplicate columns here to use in different places
        prediction_data = prediction_data.with_columns(
            pl.Series("content_preds", content)
        )
        prediction_data = prediction_data.with_columns(
            pl.Series("wording_preds", wording)
        )

        return {"data_with_predictions": prediction_data}
