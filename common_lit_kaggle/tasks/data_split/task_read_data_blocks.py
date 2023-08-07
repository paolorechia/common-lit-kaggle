from typing import Any, Mapping

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.tables import EvalSplitTable, TrainSplitTable

from .data_blocks import create_data_blocks


class ReadEvalDataBlocksTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = table_io.read_table(EvalSplitTable())
        data_blocks = create_data_blocks(input_data)
        return {"eval_data": input_data, "eval_data_blocks": data_blocks}


class ReadTrainDataBlocksTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = table_io.read_table(TrainSplitTable())
        data_blocks = create_data_blocks(input_data)
        return {"train_data": input_data, "train_data_blocks": data_blocks}
