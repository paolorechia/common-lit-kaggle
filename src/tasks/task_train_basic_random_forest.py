from typing import Any, Mapping

import mlflow
import polars as pl
from sklearn.ensemble import RandomForestRegressor

from framework.task import Task


class TrainBasicRandomForestTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        train_data: pl.DataFrame = context["enriched_train_data"]

        # Get features

        x_features = train_data.select(
            "text_length", "word_count", "sentence_count", "unique_words"
        ).to_numpy()

        # Get wording labels
        y_wording = train_data.select("wording").to_numpy()

        # Get content labels
        y_content = train_data.select("content").to_numpy()

        wording_regressor = RandomForestRegressor()
        content_regressor = RandomForestRegressor()

        wording_regressor.fit(x_features, y_wording)

        content_regressor.fit(x_features, y_content)

        return {}
