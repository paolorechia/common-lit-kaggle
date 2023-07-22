import polars as pl

from framework.task import Task
from settings import config


class SplitTrainTestTask(Task):
    def run(self):
        input_data = pl.read_csv(config.INPUT_CSV_FULL_PATH)
        print(input_data)
