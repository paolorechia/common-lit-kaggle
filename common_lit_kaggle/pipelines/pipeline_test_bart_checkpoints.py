from common_lit_kaggle import tasks
from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.settings.config import Config


class TestBartCheckpoints(Pipeline):
    def __init__(self) -> None:
        config = Config.get()
        predict_prepare_tensor_data = tasks.PrepareTensorPredictDataTask()
        predict_prepare_tensor_data.set_string_length_truncation(
            config.string_truncation_length
        )

        super().__init__(
            "test_bert_checkpoints",
            [
                tasks.ReadTestDataTask(),
                tasks.CreateUnifiedTextTestDataTask(),
                predict_prepare_tensor_data,
                tasks.TestBartCheckpointsTask(),
            ],
        )
