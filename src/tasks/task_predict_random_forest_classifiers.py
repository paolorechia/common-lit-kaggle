from typing import Any, Mapping

import mlflow
import polars as pl
from sklearn.ensemble import RandomForestRegressor

from framework.task import Task


class PredictBasicRandomForestTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        prediction_data: pl.DataFrame = context["enriched_prediction_data"]

        wording_regressor: RandomForestRegressor = context["wording_regressor"]
        content_regressor: RandomForestRegressor = context["content_regressor"]
        mlflow.autolog()

        x_features = prediction_data.select(
            "text_length", "word_count", "sentence_count", "unique_words"
        ).to_numpy()

        wording_preds = wording_regressor.predict(x_features)

        content_preds = content_regressor.predict(x_features)
        prediction_data = prediction_data.with_columns(
            pl.Series("wording", wording_preds)
        )
        prediction_data = prediction_data.with_columns(
            pl.Series("content", content_preds)
        )

        return {"data_with_predictions": prediction_data}
