"""Train code adapted from PyTorch tutorial:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

"""
import logging
from typing import Any, Mapping, Optional

from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoConfig, DebertaV2Config, DebertaV2Model

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.modeling import DebertaTwinsWithRegressionHead, train_model
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.utils.mlflow_wrapper import mlflow

logger = logging.getLogger(__name__)

# pylint: disable=no-member,too-many-ancestors
# pylint: disable=invalid-name,consider-using-f-string


class TrainDebertaTwinsFromCheckpointTask(Task):
    def __init__(self, name: str | None = None, checkpoint_path=None) -> None:
        super().__init__(name)
        self.truncation_length: Optional[int] = None
        self.checkpoint_path = checkpoint_path
        assert self.checkpoint_path, "Must give a checkpoint to resume task!"

    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        train_data = context["tensor_train_data"]
        eval_data = context["tensor_eval_data"]
        checkpoint_model = context["checkpoint_path"]

        config = Config.get()

        # Set specific configuration for resume trial trainig
        # We want a lower learning rate here
        config.learning_rate = 0.0001
        config.step_lr_step_size = 1
        config.step_lr_gamma = 0.9

        deberta_config: DebertaV2Config = AutoConfig.from_pretrained(
            config.model_custom_config_dir
        )
        logger.info("Loaded the following config: %s", deberta_config)

        deberta_twins = DebertaTwinsWithRegressionHead.from_checkpoint(
            checkpoint_model,
            config=deberta_config,
            freeze_prompt=True,
            freeze_answer=False,
            freeze_poolers=False,
        )

        deberta_twins.to(config.device)
        deberta_twins.train(True)
        train_sampler = RandomSampler(train_data)

        batch_size = config.batch_size
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=batch_size
        )

        eval_sampler = RandomSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=batch_size
        )

        train_model(
            train_dataloader,
            deberta_twins,
            eval_dataloader=eval_dataloader,
        )

        model_name = config.model.replace("/", "-")
        deberta_twins.save_pretrained(f"trained_{model_name}")
        return {"trained_bart_path": "trained_bart", "bart_model": deberta_twins}
