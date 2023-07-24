from typing import Any, Mapping

import polars as pl
from sklearn.ensemble import RandomForestRegressor

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config


class TrainBasicRandomForestTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        config = Config.get()
        train_data: pl.DataFrame = context["enriched_train_data"]

        # Get features
        try:
            extra_features = context["extra_features"]
        except KeyError:
            extra_features = None

        features = config.used_features

        if extra_features:
            features.extend(extra_features)

        x_features = train_data.select(features).to_numpy()

        # Get wording labels
        y_wording = train_data.select("wording").to_numpy()

        # Get content labels
        y_content = train_data.select("content").to_numpy()

        content_regressor = RandomForestRegressor()

        content_regressor.fit(x_features, y_content)

        wording_regressor = RandomForestRegressor()
        wording_regressor.fit(x_features, y_wording)

        return {
            "wording_regressor": wording_regressor,
            "content_regressor": content_regressor,
            "features": features,
        }
