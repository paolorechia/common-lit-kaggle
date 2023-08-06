from .task_join_input_data import JoinInputTask
from .task_read_augmented_data import (
    MergeAugmentedSourcesTask,
    ReadBertTrainTask,
    ReadGPT2TrainTask,
    ReadPPDBTrainTask,
    ReadT5TrainTask,
    ReadWMT19TrainTask,
    ReadWord2VecTrainTask,
)
from .task_read_data_blocks import ReadEvalDataBlocksTask, ReadTrainDataBlocksTask
from .task_read_eval_data import ReadEvalDataTask
from .task_read_input_data import ReadInputDataTask
from .task_read_prediction_input_data import ReadPredictionInputDataTask
from .task_read_prediction_prompt_data import ReadPredictionInputPromptDataTask
from .task_read_prompt_data import ReadInputPromptDataTask
from .task_read_test_data import ReadTestDataTask
from .task_read_train_data import ReadTrainDataTask
from .task_split_train_test import SplitTrainTestByPromptTask
from .task_write_predictions import WritePredictionsTask
