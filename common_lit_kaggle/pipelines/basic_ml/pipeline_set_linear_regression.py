from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tasks import basic_ml, data_split
from common_lit_kaggle.utils.mlflow_wrapper import mlflow

LABEL = "set_train_linear"


class SentenceTransformerLinearRegressionPipeline(Pipeline):
    def __init__(self) -> None:
        config = Config.get()

        for idx, feature in enumerate(config.used_features):
            mlflow.log_param(f"features_{idx}", feature)

        for idx, prompt in enumerate(config.train_prompts):
            mlflow.log_param(f"train_prompt_{idx}", prompt)

        mlflow.log_param("distance_metric", config.distance_metric)
        mlflow.log_param("sentence_transformer", config.sentence_transformer)
        mlflow.log_param("distance_stategy", config.distance_stategy)

        super().__init__(
            LABEL,
            [
                # Train
                data_split.ReadTrainDataTask(),
                basic_ml.AddBasicFeaturesTrainTask(),
                basic_ml.AddSentenceEmbeddingToTrainTask(),
                basic_ml.TrainBasicLinearRegressorTask(),
                # Test
                data_split.ReadTestDataTask(),
                basic_ml.AddBasicFeaturesTestTask(),
                basic_ml.AddSentenceEmbeddingToTestTask(),
                basic_ml.TestBasicLinearRegressorTask(),
                basic_ml.AnalysePredictionsTask(name=LABEL),
            ],
        )
