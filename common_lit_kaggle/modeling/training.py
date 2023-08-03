import logging
from typing import Optional, Union

try:
    import bitsandbytes as bnb
except ImportError:
    print("Could not import bitsandbytes! This is currently expected in Kaggle")

import numpy as np
from torch import nn, optim
from tqdm import tqdm

from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.utils.checkpoint import get_checkpoint_path
from common_lit_kaggle.utils.mlflow_wrapper import mlflow

from .bart import BartWithRegressionHead

# from .falcon import FalconLoraWithRegressionHead

# pylint: disable=no-member,too-many-ancestors
# pylint: disable=invalid-name,consider-using-f-string

logger = logging.getLogger(__name__)


class EarlyStopper:
    """Code from: https://stackoverflow.com/a/73704579/8628527"""

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
def train_epoch(
    dataloader,
    model: BartWithRegressionHead,
    optimizer,
    scheduler,
    criterion,
    weights=None,  
):
    """Adapted from: https://huggingface.co/docs/transformers/v4.26.1/training#training-loop"""
    total_loss = 0
    config = Config.get()

    idx = 1
    for data in tqdm(dataloader):
        input_tensor, target_tensor = data
        logits = model.forward(input_tensor)

        # Compute weights for the current batch of data
        if weights is not None:
            weights_batch = compute_instance_weights(target_tensor)  

        # Compute loss here
        if weights is not None:
            loss = (weights_batch * (logits - target_tensor) ** 2).mean() / config.gradient_accumulation_steps
        else:
            loss = criterion(logits, target_tensor) / config.gradient_accumulation_steps

        loss.backward()

        if idx % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config.gradient_accumulation_steps
        idx += 1

    scheduler.step()
    return total_loss / len(dataloader)

def eval_epoch(
    dataloader,
    model: Union[BartWithRegressionHead],
    criterion,
    weights=None,  
):
    total_loss = 0

    for data in tqdm(dataloader):
        input_tensor, target_tensor = data
        logits = model.forward(input_tensor)

        # Compute weights for the current batch of data
        if weights is not None:
            weights_batch = compute_instance_weights(target_tensor)  

        # Compute loss here
        if weights is not None:
            loss = (weights_batch * (logits - target_tensor) ** 2).mean()
        else:
            loss = criterion(logits, target_tensor)

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(
    train_dataloader,
    model: BartWithRegressionHead,
    print_every=1,
    eval_dataloader=None,
    early_stopper: Optional[EarlyStopper] = None,
    use_8bit_optimizer=False,
    instance_weights=None, 
):
    config = Config.get()

    print_loss_total = 0  

    if use_8bit_optimizer:
        optimizer = bnb.optim.AdamW(
            model.parameters(), lr=config.learning_rate, optim_bits=8
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    print(model.parameters())
    criterion = nn.MSELoss()

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config.step_lr_step_size, gamma=config.step_lr_gamma
    )

    mlflow.log_params(
        {
            "step_lr_step_size": config.step_lr_step_size,
            "step_lr_gamma": config.step_lr_gamma,
        }
    )

    if early_stopper:
        assert eval_dataloader, "To use early stopper we need an eval dataloader!"

    should_stop = False

    for epoch in range(1, config.num_train_epochs + 1):
        model.train()

        logger.info("Starting epoch: %d", epoch)
        # Pass the instance weights to the train_epoch function
        loss = train_epoch(train_dataloader, model, optimizer, scheduler, criterion, instance_weights)
        print_loss_total += loss


        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            logger.info(
                "TRAIN LOSS: %.4f",
                print_loss_avg,
            )

        mlflow.log_metric("train_loss", print_loss_avg, epoch)

        eval_loss = None
        if eval_dataloader:
            logger.info("Evaluating on validation dataset")
            model.eval()
            # Pass the instance weights to the eval_epoch function
            eval_loss = eval_epoch(eval_dataloader, model, criterion, instance_weights)
            logger.info("EVAL LOSS: %.4f", eval_loss)

        if early_stopper:
            assert eval_loss, "Cannot use early stopper without eval loss!"
            should_stop = early_stopper.early_stop(eval_loss)

        if config.save_checkpoints:
            checkpoint_path = get_checkpoint_path(epoch=epoch)
            logger.info(
                "Saving checkpoint for epoch %d at '%s'", epoch, checkpoint_path
            )
            model.save_pretrained(checkpoint_path)

        if should_stop:
            logger.info("Training stopped early!")
            mlflow.log_metric("early_stop", True)
            return

    mlflow.log_metric("early_stop", False)
