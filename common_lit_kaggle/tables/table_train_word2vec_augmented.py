import pathlib

from common_lit_kaggle.framework.table import TableReference
from common_lit_kaggle.schemas import AugmentedTrainSplit
from common_lit_kaggle.settings.config import Config


class AugmentedWord2VecTrainTable(TableReference):
    def __init__(self):
        config = Config.get()
        super().__init__(
            name="train_word_vec_augmented",
            path=pathlib.Path(
                config.data_train_dir, "train_word2vec_augmented_split.csv"
            ),
            schema=AugmentedTrainSplit,
            format="csv",
        )
