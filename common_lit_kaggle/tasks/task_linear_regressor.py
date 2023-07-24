from typing import Any, Mapping

import polars as pl
from sklearn.linear_model import LinearRegression

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config


class TrainBasicLinearRegressorTask(Task):
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

        content_regressor = LinearRegression()

        content_regressor.fit(x_features, y_content + 3)

        wording_regressor = LinearRegression()
        wording_regressor.fit(x_features, y_wording + 3)

        return {
            "wording_regressor": wording_regressor,
            "content_regressor": content_regressor,
            "features": features,
        }


class TestBasicLinearRegressorTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        test_data: pl.DataFrame = context["enriched_test_data"]
        original_test_data: pl.DataFrame = context["test_data"]

        wording_regressor: LinearRegression = context["wording_regressor"]
        content_regressor: LinearRegression = context["content_regressor"]

        used_features = context["features"]

        x_features = test_data.select(used_features).to_numpy()

        wording_preds = wording_regressor.predict(x_features)

        content_preds = content_regressor.predict(x_features)

        data_with_predictions = original_test_data.with_columns(
            pl.Series(name="wording_preds", values=wording_preds - 3),
            pl.Series(name="content_preds", values=content_preds - 3),
        )

        return {"data_with_predictions": data_with_predictions}
