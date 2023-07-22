import argparse

import pipelines
from framework import build_pipeline_registry
from utils.setup import create_stdout_handler

create_stdout_handler()

pipeline_registry = build_pipeline_registry(pipelines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pipeline executor")
    parser.add_argument("pipeline_name")

    args = parser.parse_args()

    pipeline_name: str = args.pipeline_name

    fetched_pipe = pipeline_registry.get(pipeline_name, None)
    if fetched_pipe is None:
        raise KeyError(f"Requested pipeline does not exist: {pipeline_name}")

    fetched_pipe.run()
