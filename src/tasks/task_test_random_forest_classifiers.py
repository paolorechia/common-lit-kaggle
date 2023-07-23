from typing import Any, Mapping

import mlflow
import polars as pl
from sklearn.ensemble import RandomForestRegressor

from framework.task import Task


class TestBasicRandomForestTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        test_data: pl.DataFrame = context["enriched_test_data"]
        wording_regressor: RandomForestRegressor = context["wording_regressor"]
        content_regressor: RandomForestRegressor = context["content_regressor"]
        mlflow.autolog()

        # Get wording labels
        y_wording = test_data.select("wording").to_numpy()

        # Get content labels
        y_content = test_data.select("content").to_numpy()

        x_features = test_data.select(
            "text_length", "word_count", "sentence_count", "unique_words"
        ).to_numpy()

        score = wording_regressor.score(x_features, y_wording)
        print("Wording score", score)
        score = content_regressor.score(x_features, y_content)
        print("Content score", score)

        return {}
