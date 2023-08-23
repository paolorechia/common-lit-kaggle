import json

from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tasks import bart_twins, data_split, deberta_twins
from common_lit_kaggle.utils.mlflow_wrapper import mlflow


class ResumeDebertaTwinsRegressionPipeline(Pipeline):
    def __init__(self) -> None:
        config = Config.get()

        mlflow.set_tag("model_name", "resumed_deberta_xsmall_twin")
        mlflow.log_params(
            {
                "twins": True,
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
            "resume_deberta_twins",
            [
                # Load training data
                data_split.ReadTrainDataTask(),
                bart_twins.PrepareTensorTrainTwinsDataTask(
                    truncation_length=config.string_truncation_length
                ),
                # Load eval data
                data_split.ReadEvalDataTask(),
                bart_twins.PrepareTensorTrainTwinsDataTask(
                    truncation_length=config.string_truncation_length,
                    unified_text_data_key="eval_data",
                    output_text_data_key="tensor_eval_data",
                ),
                # Train
                deberta_twins.TrainDebertaTwinsFromCheckpointTask(
                    checkpoint_path="/home/paolo/kaggle/common-lit-kaggle/data/checkpoints/trained_microsoft-deberta-v3-xsmall_378caa44000b428dbab44ac0b05a492a_10"
                ),
            ],
        )
