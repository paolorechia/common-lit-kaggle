"""Train code adapted from PyTorch tutorial:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

"""
import logging
from typing import Any, Mapping, Optional

from torch.utils.data import DataLoader, RandomSampler

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.modeling import BartWithRegressionHead, train_model
from common_lit_kaggle.settings.config import Config

logger = logging.getLogger(__name__)

# pylint: disable=no-member,too-many-ancestors
# pylint: disable=invalid-name,consider-using-f-string


class TrainBartTask(Task):
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

        bart_model = BartWithRegressionHead.from_pretrained(model_path)
        bart_model.to(config.device)

        bart_model.train(True)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=batch_size
        )
        train_model(
            train_dataloader,
            bart_model,
        )
        bart_model.save_pretrained("trained_bart")
        return {"trained_bart_path": "trained_bart"}
