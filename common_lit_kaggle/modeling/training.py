import logging
import time

from torch import nn, optim
from tqdm import tqdm

from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.utils.checkpoint import get_checkpoint_path
from common_lit_kaggle.utils.mlflow_wrapper import mlflow

from .bart import BartWithRegressionHead

# pylint: disable=no-member,too-many-ancestors
# pylint: disable=invalid-name,consider-using-f-string

logger = logging.getLogger(__name__)


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


def eval_epoch(dataloader, model: BartWithRegressionHead, criterion):
    total_loss = 0
    for data in tqdm(dataloader):
        input_tensor, target_tensor = data
        logits = model.forward(input_tensor)

        # Compute loss here
        loss = criterion(logits, target_tensor)
        # lr_scheduler.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(
    train_dataloader,
    model,
    print_every=1,
    eval_dataloader=None,
):
    config = Config.get()

    print_loss_total = 0  # Reset every print_every

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(1, config.num_train_epochs + 1):
        logger.info("Starting epoch: %d", epoch)
        loss = train_epoch(train_dataloader, model, optimizer, criterion)
        print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            logger.info(
                "TRAIN LOSS: %.4f",
                print_loss_avg,
            )

        mlflow.log_metric("train_loss", print_loss_avg, epoch - 1)

        if eval_dataloader:
            logger.info("Evaluating on validation dataset")
            model.eval()
            # Validate model
            eval_loss = eval_epoch(eval_dataloader, model, criterion)
            logger.info("EVAL LOSS: %.4f", eval_loss)

            mlflow.log_metric("eval_loss", eval_loss)
            model.train()

        if config.save_checkpoints:
            checkpoint_path = get_checkpoint_path()
            logger.info(
                "Saving checkpoint for epoch %d at '%s'", epoch, checkpoint_path
            )
            model.save_pretrained(checkpoint_path)
