import logging
import os

import polars as pl

from common_lit_kaggle.framework.table import TableReference

logger = logging.getLogger(__name__)


def _validate_schema(polars_dataframe: pl.DataFrame, table: TableReference):
    schema_dict = table.schema.to_dict()

    for column in polars_dataframe.columns:
        column_type = polars_dataframe[column].dtype
        # logger.info("Found column: %s (type: %s)", column, column_type)
        expected_col_type = schema_dict[column]
        assert (
            expected_col_type == column_type
        ), f"Column type mismatch. Expected: {expected_col_type}, found: {column_type}"


def read_table(table: TableReference) -> pl.DataFrame:
    logger.info("Reading table: %s", table)
    polars_dataframe = None
    if table.format == "csv":
        polars_dataframe = pl.read_csv(table.path)
        _validate_schema(polars_dataframe, table)

    assert (
        polars_dataframe is not None
    ), f"Invalid or unsupported format: {table.format}"
    return polars_dataframe


def write_table(dataframe: pl.DataFrame, table: TableReference):
    if table.format == "csv":
        _validate_schema(dataframe, table)
        try:
            os.makedirs(table.path.parent)
        except FileExistsError:
            pass
        dataframe.write_csv(table.path)
    else:
        raise TypeError(f"Invalid or unsupported format: {table.format}")
