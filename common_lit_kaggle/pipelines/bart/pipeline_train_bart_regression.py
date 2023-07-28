from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tasks import bart, basic_ml, data_split


class TrainBartRegressionPipeline(Pipeline):
    def __init__(self) -> None:
        prepare_tensor_data = bart.PrepareTensorTrainDataTask()
        config = Config.get()
        # Bart supports up to 1024 sub-word tokens
        # Let's consider only the last 4096 characters
        # This should guarantee (if this approximation is correct)
        # That we truncate only prompt data instead of student data
        prepare_tensor_data.set_string_length_truncation(
            config.string_truncation_length
        )

        predict_prepare_tensor_data = bart.PrepareTensorPredictDataTask()
        predict_prepare_tensor_data.set_string_length_truncation(
            config.string_truncation_length
        )

        super().__init__(
            "train_bart_regression",
            [
                data_split.ReadTrainDataTask(),
                bart.CreateUnifiedTextTrainDataTask(),
                bart.ExploreUnifiedInputDataTask(),
                prepare_tensor_data,
                bart.TrainBartTask(),
                data_split.ReadTestDataTask(),
                bart.CreateUnifiedTextTestDataTask(),
                predict_prepare_tensor_data,
                bart.PredictBertTask(),
                data_split.WritePredictionsTask(),
                basic_ml.AnalysePredictionsTask(),
            ],
        )
