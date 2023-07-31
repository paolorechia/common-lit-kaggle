"""Train code adapted from PyTorch tutorial:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

"""
import logging
from typing import Any, Mapping, Optional

from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoConfig, BartConfig

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.modeling import BartWithRegressionHead, EarlyStopper, train_model
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.utils.mlflow_wrapper import mlflow

logger = logging.getLogger(__name__)

# pylint: disable=no-member,too-many-ancestors
# pylint: disable=invalid-name,consider-using-f-string


class TrainBartTask(Task):
    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.truncation_length: Optional[int] = None

    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        train_data = context["tensor_train_data"]
        eval_data = context["tensor_eval_data"]

        config = Config.get()

        model_path = config.bart_model
        batch_size = config.batch_size

        bart_config: BartConfig = AutoConfig.from_pretrained(
            config.model_custom_config_dir
        )

        logger.info("Loaded the following config: %s", bart_config)

        for key in dir(bart_config):
            if "dropout" in key:
                mlflow.log_param(f"bart_{key}", getattr(bart_config, key))

        bart_model = BartWithRegressionHead.from_pretrained(
            model_path, config=bart_config
        )

        bart_model.to(config.device)

        bart_model.train(True)
        train_sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=batch_size
        )

        eval_sampler = RandomSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=batch_size
        )
        early_stopper = EarlyStopper(
            patience=config.early_stop_patience, min_delta=config.early_stop_min_delta
        )

        train_model(
            train_dataloader,
            bart_model,
            eval_dataloader=eval_dataloader,
            early_stopper=early_stopper,
        )

        model_name = config.bart_model.replace("/", "-")
        bart_model.save_pretrained(f"trained_{model_name}")
        return {"trained_bart_path": "trained_bart", "bart_model": bart_model}
