import polars as pl

from common_lit_kaggle.framework.schema import Schema


class InputPromptSchema(Schema):
    prompt_id = pl.Utf8
    prompt_question = pl.Utf8
    prompt_title = pl.Utf8
    prompt_text = pl.Utf8
