import json

from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tasks import bart, data_split
from common_lit_kaggle.utils.mlflow_wrapper import mlflow


class TrainBartWithPPDBAugmentationPipeline(Pipeline):
    def __init__(self) -> None:
        config = Config.get()
        mlflow.set_tags({"name": config.model, "preprocessing": "ppdb_augmentation"})
        mlflow.log_params(
            {
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
                "undersampling_multiplier": config.min_count_multiplier,
            }
        )

        super().__init__(
            "train_bart_ppdb",
            [
                # Load training data
                data_split.ReadTrainDataTask(),
                data_split.ReadPPDBTrainTask(),
                data_split.MergeAugmentedSourcesTask(
                    data_sources=["ppdb_augmented_train_data"]
                ),
                bart.CreateUnifiedTextTrainDataTask(),
                bart.ExploreUnifiedInputDataTask(),
                bart.PrepareTensorTrainDataTask(
                    truncation_length=config.string_truncation_length
                ),
                # Load eval data
                data_split.ReadEvalDataTask(),
                bart.CreateUnifiedTextEvalDataTask(),
                bart.PrepareTensorTrainDataTask(
                    truncation_length=config.string_truncation_length,
                    unified_text_data_key="eval_unified_text_data",
                    output_text_data_key="tensor_eval_data",
                ),
                # Train
                bart.TrainBartTask(),
            ],
        )
