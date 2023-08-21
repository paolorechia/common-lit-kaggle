import json

from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tasks import data_split, reinforcement_learning
from common_lit_kaggle.utils.mlflow_wrapper import mlflow


class RLGPT2(Pipeline):
    def __init__(self) -> None:
        config = Config.get()
        mlflow.set_tags({"name": config.model})
        mlflow.log_params(
            {
                "randomized_label_func": "1.0 + (3 * random.random())",
                "cost_sensitive_learning": config.cost_sensitive_learning,
                "cost_sensitive_multiplier": config.cost_sensitive_exponent,
                "cost_sensitive_sum_operand": config.cost_sensitive_sum_operand,
                "regression_dropout": config.dropout,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "truncation_length": config.string_truncation_length,
                "model_context_length": config.model_context_length,
                "gradient_acumulation_steps": config.gradient_accumulation_steps,
                "eval_prompts": json.dumps(config.eval_prompts),
                "test_prompts": json.dumps(config.test_prompts),
                "virtual_batch_size": config.batch_size
                * config.gradient_accumulation_steps,
            }
        )

        super().__init__(
            "rl_gpt2",
            [
                # Load training data
                data_split.ReadTrainDataTask(),
                reinforcement_learning.RLGPT2Task(),
            ],
        )
