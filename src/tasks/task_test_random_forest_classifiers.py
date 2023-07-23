from typing import Any, Mapping

import mlflow
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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

        wording_preds = wording_regressor.predict(x_features)

        content_preds = content_regressor.predict(x_features)

        score = mean_squared_error(wording_preds, y_wording, squared=True)
        print("Wording score", score)
        score = mean_squared_error(content_preds, y_content, squared=True)
        print("Content score", score)

        return {}
