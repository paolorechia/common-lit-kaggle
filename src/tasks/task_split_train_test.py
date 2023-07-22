from framework import table_io
from framework.task import Task
from settings import config
from tables.summaries import InputSummaries


class SplitTrainTestTask(Task):
    def run(self):
        input_data = table_io.read_table(InputSummaries())
        print(input_data)
