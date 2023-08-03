import logging
from typing import Optional, Union
import wandb
try:
    import bitsandbytes as bnb
except ImportError:
    print("Could not import bitsandbytes! This is currently expected in Kaggle")

import numpy as np
import torch
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
    input_tensor, 
    content_targets,
    wording_targets,
    model: BartWithRegressionHead,
    optimizer,
    scheduler,
    criterion,
    content_weights,
    wording_weights
):
    model.zero_grad()
    model_output = model(input_tensor)
    content_output, wording_output = model_output[:, 0], model_output[:, 1]

    # Compute weighted loss
    content_loss = criterion(content_output, content_targets)
    content_loss = (content_loss * content_weights).mean()
    
    wording_loss = criterion(wording_output, wording_targets)
    wording_loss = (wording_loss * wording_weights).mean()

    loss = (content_loss + wording_loss) / 2
    loss.backward()

    optimizer.step()
    scheduler.step()

    return loss.item()

def eval_epoch(
    input_tensor,
    content_targets,
    wording_targets,
    model: BartWithRegressionHead,
    criterion,
    content_weights,
    wording_weights
):
    with torch.no_grad():
        model_output = model(input_tensor)
        content_output, wording_output = model_output[:, 0], model_output[:, 1]

        # Compute weighted loss
        content_loss = criterion(content_output, content_targets)
        content_loss = (content_loss * content_weights).mean()

        wording_loss = criterion(wording_output, wording_targets)
        wording_loss = (wording_loss * wording_weights).mean()

        loss = (content_loss + wording_loss) / 2

    return loss.item()


def train_model(
    train_dataloader,
    model: BartWithRegressionHead,
    print_every=1,
    eval_dataloader=None,
    early_stopper: Optional[EarlyStopper] = None,
    use_8bit_optimizer=False,
    content_threshold=1.0,
    wording_threshold=1.0
):
    config = Config.get()
    
    print_loss_total = 0  # Reset every print_every
    step = 0
    wandb.init(project='bart-training-4') # You should define the project name
    wandb.watch(model, log_freq=10)
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

        # Add weights calculation and pass them to train_epoch function
        for batch in train_dataloader:
            input_tensor, targets = batch
            content_targets, wording_targets = targets[:, 0], targets[:, 1]

            model_output = model(input_tensor)
            content_output, wording_output = model_output[:, 0], model_output[:, 1]

            content_weights = (torch.abs(content_targets - content_output) > content_threshold).float()
            wording_weights = (torch.abs(wording_targets - wording_output) > wording_threshold).float()

            loss = train_epoch(input_tensor, content_targets, wording_targets, model, optimizer, scheduler, criterion, content_weights, wording_weights)
            print_loss_total += loss
            if step % 10 == 0: # Log the loss every 10 steps
                wandb.log({"train_loss": loss})
            step += 1

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
            # Validate model
            for batch in eval_dataloader:
                input_tensor, targets = batch
                content_targets, wording_targets = targets[:, 0], targets[:, 1]

                model_output = model(input_tensor)
                content_output, wording_output = model_output[:, 0], model_output[:, 1]

                content_weights = (torch.abs(content_targets - content_output) > content_threshold).float()
                wording_weights = (torch.abs(wording_targets - wording_output) > wording_threshold).float()

                eval_loss = eval_epoch(input_tensor, content_targets, wording_targets, model, criterion, content_weights, wording_weights)

            logger.info("EVAL LOSS: %.4f", eval_loss)

            wandb.log({"eval_loss": eval_loss, "epoch": epoch})  # Log eval loss to wandb
            model.train()

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
