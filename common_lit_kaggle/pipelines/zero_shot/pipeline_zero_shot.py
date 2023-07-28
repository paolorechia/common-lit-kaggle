from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.tasks import basic_ml, data_split, zero_shot

LABEL = "zero_train_random_forest"


class ZeroShotRandomForestPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            LABEL,
            [
                data_split.ReadTrainDataTask(),
                basic_ml.AddBasicFeaturesTrainTask(),
                zero_shot.AddZeroShotLabelTrainTask(),
                basic_ml.TrainBasicRandomForestTask(),
                data_split.ReadTestDataTask(),
                basic_ml.AddBasicFeaturesTestTask(),
                basic_ml.TestBasicRandomForestTask(name=LABEL),
            ],
        )
