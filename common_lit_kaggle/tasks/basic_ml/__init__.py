from .task_add_basic_features_prediction import AddBasicFeaturesPredictionTask
from .task_add_basic_features_test import AddBasicFeaturesTestTask
from .task_add_basic_features_train import AddBasicFeaturesTrainTask
from .task_add_sentence_embedding import (
    AddSentenceEmbeddingToPredictTask,
    AddSentenceEmbeddingToTestTask,
    AddSentenceEmbeddingToTrainTask,
)
from .task_analyse_predictions import AnalysePredictionsTask
from .task_explore_input_data import ExploreInputDataTask
from .task_linear_regressor import (
    TestBasicLinearRegressorTask,
    TrainBasicLinearRegressorTask,
)
from .task_predict_random_forest_classifiers import PredictBasicRandomForestTask
from .task_test_random_forest_classifiers import TestBasicRandomForestTask
from .task_train_basic_random_forest import TrainBasicRandomForestTask
