from common_lit_kaggle import tables
from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tasks import data_balancing


class PlotAugmentedPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            "plot_augmented",
            [
                data_balancing.TaskPlotAugmented(tables.AugmentedBertTrainTable()),
                data_balancing.TaskPlotAugmented(tables.AugmentedGPT2TrainTable()),
                data_balancing.TaskPlotAugmented(tables.AugmentedPPDBTrainTable()),
                data_balancing.TaskPlotAugmented(tables.AugmentedT5TrainTable()),
                data_balancing.TaskPlotAugmented(tables.AugmentedWmt19TrainTable()),
                data_balancing.TaskPlotAugmented(tables.AugmentedWord2VecTrainTable()),
            ],
        )
