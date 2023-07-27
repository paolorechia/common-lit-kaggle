"""Train code adapted from PyTorch tutorial:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

"""
import logging
import math
import time
from typing import Any, Mapping, Optional

from torch.nn import NLLLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from transformers import BartModel

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config

logger = logging.getLogger(__name__)


# pylint: disable=invalid-name, consider-using-f-string
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


def train_epoch(dataloader, model, optimizer, criterion):
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        optimizer.zero_grad()

        model_output = model.forward(input_tensor)

        loss = criterion(
            model_output.view(-1, model_output.size(-1)), target_tensor.view(-1)
        )
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(
    train_dataloader,
    model,
    n_epochs,
    learning_rate=0.001,
    print_every=100,
    plot_every=100,
):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # TODO: We probably want to change the optimizer and loss here
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, model, optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

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

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


class TrainBertTask(Task):
    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.truncation_length: Optional[int] = None

    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        train_data = context["tensor_train_data"]
        config = Config.get()

        model_path = config.bart_model
        num_epochs = config.num_train_epochs
        batch_size = config.batch_size
        learning_rate = config.learning_rate

        bart_model = BartModel.from_pretrained(model_path)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=batch_size
        )
        train(
            train_dataloader,
            bart_model,
            num_epochs,
            learning_rate,
        )
        return {}
