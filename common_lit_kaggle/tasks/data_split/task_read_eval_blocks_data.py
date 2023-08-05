from typing import Any, Mapping

import polars as pl

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.tables import EvalSplitTable


class ReadEvalDataBlocksTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = table_io.read_table(EvalSplitTable())
        ranges = [(-5, -1), (-1, 0), (0, 1), (1, 2), (2, 3), (4, 10)]

        data_blocks: dict[str, list[pl.DataFrame]] = {"content": [], "wording": []}
        for attr in ["content", "wording"]:
            for range_ in ranges:
                block_data = input_data.select(
                    input_data.filter(
                        pl.col(attr) > range_[0] and pl.col(attr) < range_[1]
                    )
                )
                data_blocks[attr].append(block_data)

        return {"eval_data": input_data, "eval_data_blocks": data_blocks}
