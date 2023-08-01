"""Train code adapted from PyTorch tutorial:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

"""
import logging
from typing import Any, Mapping, Optional

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoConfig, AutoModel, FalconConfig

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.modeling import (
    EarlyStopper,
    FalconLoraWithRegressionHead,
    train_model,
)
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.utils.mlflow_wrapper import mlflow

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ],
)

logger = logging.getLogger(__name__)

# pylint: disable=no-member,too-many-ancestors
# pylint: disable=invalid-name,consider-using-f-string


class TrainFalconTask(Task):
    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.truncation_length: Optional[int] = None

    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        train_data = context["tensor_train_data"]
        eval_data = context["tensor_eval_data"]

        config = Config.get()

        model_path = config.model
        assert "falcon" in model_path, "This task expects a falcon model"

        batch_size = config.batch_size

        falcon_config: FalconConfig = AutoConfig.from_pretrained(
            config.model_custom_config_dir,
        )

        logger.info("Loaded the following config: %s", falcon_config)

        for key in dir(falcon_config):
            if "dropout" in key:
                mlflow.log_param(f"bart_{key}", getattr(falcon_config, key))

        falcon_model = AutoModel.from_pretrained(
            config.model, config=falcon_config, load_in_8bit=True
        )

        falcon_model = prepare_model_for_int8_training(falcon_model)
        falcon_model = get_peft_model(falcon_model, peft_config)
        falcon_model.print_trainable_parameters()

        falcon_model = FalconLoraWithRegressionHead(
            config=falcon_config,
            falcon_model=falcon_model,
        )

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
            falcon_model,
            eval_dataloader=eval_dataloader,
            early_stopper=early_stopper,
            use_8bit_optimizer=False,
        )

        model_name = config.model.replace("/", "-")
        falcon_model.save_pretrained(f"trained_{model_name}")
        return {"trained_bart_path": "trained_bart", "bart_model": falcon_model}
