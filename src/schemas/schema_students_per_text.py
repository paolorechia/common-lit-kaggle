import polars as pl

from framework.schema import Schema


class StudentsPerTextSchema(Schema):
    describe = pl.Utf8
    student_id = pl.Utf8
    count = pl.Float64
