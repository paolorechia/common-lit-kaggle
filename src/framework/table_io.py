import logging

import polars as pl

from framework.table import TableReference

logger = logging.getLogger(__name__)


def read_table(table: TableReference) -> pl.DataFrame:
    logger.info("Reading table: %s", table)
    polars_dataframe = None
    if table.format == "csv":
        schema_dict = table.schema.to_dict()
        polars_dataframe = pl.read_csv(table.path)
        for column in polars_dataframe.columns:
            column_type = polars_dataframe[column].dtype
            logger.info("Found column: %s (type: %s)", column, column_type)
            expected_col_type = schema_dict[column]
            assert (
                expected_col_type == column_type
            ), f"Column type mismatch. Expected: {expected_col_type}, found: {column_type}"

    assert (
        polars_dataframe is not None
    ), f"Invalid or unsupported format: {table.format}"
    return polars_dataframe
