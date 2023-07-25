import polars as pl

from common_lit_kaggle.framework.schema import Schema


class UnifiedTextDataSchema(Schema):
    student_id = pl.Utf8
    prompt_id = pl.Utf8
    content = pl.Float64
    wording = pl.Float64
    unified_text = pl.Utf8
    unified_labels = pl.Utf8