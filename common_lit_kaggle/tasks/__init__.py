from .task_add_basic_features_prediction import AddBasicFeaturesPredictionTask
from .task_add_basic_features_test import AddBasicFeaturesTestTask
from .task_add_basic_features_train import AddBasicFeaturesTrainTask
from .task_add_sentence_embedding import (
    AddSentenceEmbeddingToPredictTask,
    AddSentenceEmbeddingToTestTask,
    AddSentenceEmbeddingToTrainTask,
)
from .task_add_zero_shot_label import AddZeroShotLabelTrainTask
from .task_analyse_predictions import AnalysePredictionsTask
from .task_create_unified_text_data import (
    CreateUnifiedTextTestDataTask,
    CreateUnifiedTextTrainDataTask,
)
from .task_explore_input_data import ExploreInputDataTask
from .task_explore_unified_data import ExploreUnifiedInputDataTask
from .task_join_input_data import JoinInputTask
from .task_linear_regressor import (
    TestBasicLinearRegressorTask,
    TrainBasicLinearRegressorTask,
)
from .task_predict_bart import PredictBertTask
from .task_predict_random_forest_classifiers import PredictBasicRandomForestTask
from .task_prepare_tensor_data import (
    PrepareTensorPredictDataTask,
    PrepareTensorTrainDataTask,
)
from .task_read_input_data import ReadInputDataTask
from .task_read_prediction_input_data import ReadPredictionInputDataTask
from .task_read_prediction_prompt_data import ReadPredictionInputPromptDataTask
from .task_read_prompt_data import ReadInputPromptDataTask
from .task_read_test_data import ReadTestDataTask
from .task_read_train_data import ReadTrainDataTask
from .task_split_train_test import SplitTrainTestByPromptTask
from .task_test_bart_checkpoints import TestBartCheckpointsTask
from .task_test_random_forest_classifiers import TestBasicRandomForestTask
from .task_train_bart import TrainBartTask
from .task_train_basic_random_forest import TrainBasicRandomForestTask
from .task_write_predictions import WritePredictionsTask
