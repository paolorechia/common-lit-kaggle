import logging
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.modeling import BartWithRegressionHead
from common_lit_kaggle.settings.config import Config

logger = logging.getLogger(__name__)

class TrainBartTask(Task):
    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.truncation_length: Optional[int] = None

    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        train_data = context["tensor_train_data"]
        config = Config.get()

        model_path = config.bart_model
        batch_size = config.batch_size
        gradient_accumulation_steps = config.gradient_accumulation_steps

        bart_model = BartWithRegressionHead.from_pretrained(model_path)
        bart_model.to(config.device)
        bart_model.train(True)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # Optimizer for the BART model
        optimizer = torch.optim.AdamW(bart_model.parameters(), lr=config.learning_rate)

        # Learning rate scheduler with step decay
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.num_train_epochs//10, gamma=0.1)
        
        # Mean Squared Error loss function
        criterion = nn.MSELoss()
        
        # Summary writer for TensorBoard
        writer = SummaryWriter()

        for epoch in range(config.num_train_epochs):
            running_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc="Training", leave=True)
            
            for step, batch in enumerate(progress_bar):
                inputs, labels = batch
                outputs = bart_model(inputs)

                # Custom loss computation
                loss = criterion(outputs.view(-1), labels.view(-1))
                
                # Gradient accumulation for memory efficiency
                loss = loss / gradient_accumulation_steps  
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    # Optimizer step is performed after gradient accumulation
                    optimizer.step()
                    bart_model.zero_grad()

                # Updating the running loss
                running_loss += loss.item() * inputs.size(0)
                progress_bar.set_postfix({"loss": loss.item()})

            # Scheduler step is performed after each epoch
            scheduler.step()

            # Fetching the learning rate after scheduler step for the current epoch
            current_lr = optimizer.param_groups[0]['lr']

            # Logging the loss and learning rate for visualization in TensorBoard
            writer.add_scalar("Loss/train", running_loss / len(train_dataloader.dataset), epoch)
            writer.add_scalar("LearningRate/train", current_lr, epoch)

            # Computing the epoch loss
            epoch_loss = running_loss / len(train_dataloader.dataset)
            print(f'Epoch {epoch + 1}, Train Loss: {epoch_loss}, Learning rate: {current_lr}')

            # Logging the epoch loss for visualization in TensorBoard
            writer.add_scalar("Loss/epoch", epoch_loss, epoch)

            # Saving model checkpoints every 10 epochs
            if config.save_checkpoints and epoch % 10 == 0:
                checkpoint_dir = config.checkpoints_dir / f"checkpoint_epoch_{epoch}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                bart_model.save_pretrained(checkpoint_dir)

        # Saving the final model
        bart_model.save_pretrained("trained_bart")
        writer.close()

        return {"trained_bart_path": "trained_bart", "bart_model": bart_model}
