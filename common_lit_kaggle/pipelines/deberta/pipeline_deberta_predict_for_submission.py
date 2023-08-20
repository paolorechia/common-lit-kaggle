from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tasks import bart, data_split, deberta


class DebertaPredictionRegressionPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            "predict_deberta_regression",
            [
                data_split.ReadPredictionInputDataTask(),
                bart.CreateUnifiedTextPredictionDataTask(),
                bart.PrepareTensorPredictDataTask(
                    input_data_key="prediction_unified_text_data",
                    truncation_length=Config.get().string_truncation_length,
                ),
                deberta.PredictDebertaTask(input_data_key="input_prediction_data"),
                data_split.WritePredictionsTask(),
            ],
        )
