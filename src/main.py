import argparse

import torch

from pipelines.pipeline_split_train_test import SplitTrainTestPipeline
from settings import config
from utils.setup import create_stdout_handler

create_stdout_handler()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pipeline executor")
    parser.add_argument("pipeline_name")

    args = parser.parse_args()

    pipeline_name: str = args.pipeline_name

    if pipeline_name == "split_train_test":
        SplitTrainTestPipeline().run()
