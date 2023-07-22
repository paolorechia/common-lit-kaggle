from typing import Any, Mapping

from framework import table_io
from framework.task import Task
from tables.table_prompts import InputPromptsTable


class ReadInputPromptDataTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_prompts = table_io.read_table(InputPromptsTable())
        return {"input_prompts": input_prompts}
