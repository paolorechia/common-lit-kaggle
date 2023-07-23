import polars as pl

from common_lit_kaggle.framework.schema import Schema


class OutputPredictionSchema(Schema):
    student_id = pl.Utf8
    content = pl.Float64
    wording = pl.Float64
