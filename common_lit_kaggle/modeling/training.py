import math
import time

from torch import nn, optim
from tqdm import tqdm

from common_lit_kaggle.utils.mlflow_wrapper import mlflow

from .bart import BartWithRegressionHead

# pylint: disable=no-member,too-many-ancestors
# pylint: disable=invalid-name,consider-using-f-string


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


def train_model(
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

        mlflow.log_metric("loss", print_loss_avg, epoch - 1)
        model.save_pretrained(f"trained_bart_{epoch}")
