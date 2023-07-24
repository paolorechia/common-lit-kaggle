from typing import Any, Mapping

import polars as pl
from sklearn.ensemble import RandomForestRegressor

from common_lit_kaggle.framework.task import Task


class TestBasicRandomForestTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        test_data: pl.DataFrame = context["enriched_test_data"]
        original_test_data: pl.DataFrame = context["test_data"]

        wording_regressor: RandomForestRegressor = context["wording_regressor"]
        content_regressor: RandomForestRegressor = context["content_regressor"]

        used_features = context["features"]

        x_features = test_data.select(used_features).to_numpy()

        wording_preds = wording_regressor.predict(x_features)

        content_preds = content_regressor.predict(x_features)

        data_with_predictions = original_test_data.with_columns(
            pl.Series(name="wording_preds", values=wording_preds),
            pl.Series(name="content_preds", values=content_preds),
        )

        return {"data_with_predictions": data_with_predictions}
