from common_lit_kaggle import tasks
from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.settings.config import Config


class TrainBartRegressionPipeline(Pipeline):
    def __init__(self) -> None:
        prepare_tensor_data = tasks.PrepareTensorTrainDataTask()
        config = Config.get()
        # Bart supports up to 1024 sub-word tokens
        # Let's consider only the last 4096 characters
        # This should guarantee (if this approximation is correct)
        # That we truncate only prompt data instead of student data
        prepare_tensor_data.set_string_length_truncation(
            config.string_truncation_length
        )

        predict_prepare_tensor_data = tasks.PrepareTensorPredictDataTask()
        predict_prepare_tensor_data.set_string_length_truncation(
            config.string_truncation_length
        )

        super().__init__(
            "train_bart_regression",
            [
                tasks.ReadTrainDataTask(),
                tasks.CreateUnifiedTextTrainDataTask(),
                tasks.ExploreUnifiedInputDataTask(),
                prepare_tensor_data,
                tasks.TrainBartTask(),
                tasks.ReadTestDataTask(),
                tasks.CreateUnifiedTextTestDataTask(),
                predict_prepare_tensor_data,
                tasks.PredictBertTask(),
                tasks.AnalysePredictionsTask(),
            ],
        )
