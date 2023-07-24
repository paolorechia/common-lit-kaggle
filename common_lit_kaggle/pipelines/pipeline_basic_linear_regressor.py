from common_lit_kaggle import tasks
from common_lit_kaggle.framework import Pipeline

LABEL = "basic_train_linear_regressor"


class BasicLinearRegressorPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            LABEL,
            [
                tasks.ReadTrainDataTask(),
                tasks.AddBasicFeaturesTrainTask(),
                tasks.TrainBasicLinearRegressorTask(),
                tasks.ReadTestDataTask(),
                tasks.AddBasicFeaturesTestTask(),
                tasks.TestBasicLinearRegressorTask(),
                tasks.WritePredictionsTask(),
                tasks.AnalysePredictionsTask(name=LABEL),
            ],
        )
