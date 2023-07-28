from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.tasks import basic_ml, data_split

LABEL = "basic_train_linear_regressor"


class BasicLinearRegressorPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            LABEL,
            [
                data_split.ReadTrainDataTask(),
                basic_ml.AddBasicFeaturesTrainTask(),
                basic_ml.TrainBasicLinearRegressorTask(),
                data_split.ReadTestDataTask(),
                basic_ml.AddBasicFeaturesTestTask(),
                basic_ml.TestBasicLinearRegressorTask(),
                data_split.WritePredictionsTask(),
                basic_ml.AnalysePredictionsTask(name=LABEL),
            ],
        )
