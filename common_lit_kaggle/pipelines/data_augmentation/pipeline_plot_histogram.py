from common_lit_kaggle import tables
from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tasks import data_balancing


class PlotAugmentedPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            "plot_augmented",
            [
                data_balancing.PlotAugmentedTask(tables.AugmentedBertTrainTable()),
                data_balancing.PlotAugmentedTask(tables.AugmentedGPT2TrainTable()),
                data_balancing.PlotAugmentedTask(tables.AugmentedPPDBTrainTable()),
                data_balancing.PlotAugmentedTask(tables.AugmentedT5TrainTable()),
                data_balancing.PlotAugmentedTask(tables.AugmentedWmt19TrainTable()),
                data_balancing.PlotAugmentedTask(tables.AugmentedWord2VecTrainTable()),
            ],
        )
