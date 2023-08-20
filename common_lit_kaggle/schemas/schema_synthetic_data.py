import polars as pl

from common_lit_kaggle.framework.schema import Schema


class SyntheticDataSchema(Schema):
    text = pl.Utf8
