from common_lit_kaggle import tasks
from common_lit_kaggle.framework import Pipeline


class TrainBartRegressionPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            "train_bart_regression",
            [
                tasks.ReadTrainDataTask(),
                tasks.CreateUnifiedTextDataTask(),
            ],
        )
