from typing import Any, Mapping

import polars as pl

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.tables import OutputPredictionTable


class WritePredictionsTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        data_with_predictions: pl.DataFrame = context["data_with_predictions"]

        output = data_with_predictions.select(
            "student_id",
            pl.col("content_preds").alias("content"),
            pl.col("wording_preds").alias("wording"),
        )
        table_io.write_table(output, OutputPredictionTable())
        return {}
