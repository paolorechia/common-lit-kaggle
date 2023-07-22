from typing import Optional

from framework import table_io
from framework.task import Task
from tables.table_summaries import InputSummaries


class ExploreInputDataTask(Task):
    def run(self):
        input_data = table_io.read_table(InputSummaries())
        print(input_data)

        print(input_data.describe())
