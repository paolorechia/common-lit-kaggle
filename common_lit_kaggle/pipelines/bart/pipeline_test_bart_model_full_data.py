from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tasks import bart, basic_ml, data_split


class TestBartFullData(Pipeline):
    def __init__(self) -> None:
        config = Config.get()
        predict_prepare_tensor_data = bart.PrepareTensorPredictDataTask()
        predict_prepare_tensor_data.set_string_length_truncation(
            config.string_truncation_length
        )

        super().__init__(
            "test_bart_full",
            [
                data_split.ReadTestDataTask(),
                bart.CreateUnifiedTextTestDataTask(),
                predict_prepare_tensor_data,
                bart.PredictBertTask(),
                data_split.WritePredictionsTask(),
                basic_ml.AnalysePredictionsTask(),
            ],
        )
