import argparse
import textwrap

import common_lit_kaggle.pipelines as pipelines
from common_lit_kaggle.framework import build_pipeline_registry
from common_lit_kaggle.utils.setup import create_stdout_handler

if __name__ == "__main__":
    create_stdout_handler()

    pipeline_registry = build_pipeline_registry(pipelines)


    available_pipelines = list(pipeline_registry.keys())
    available_pipelines_str = f"Available pipelines: {available_pipelines}"

    parser = argparse.ArgumentParser(
        "Pipeline executor",
        epilog=textwrap.dedent(available_pipelines_str)
    )
    

    parser.add_argument("pipeline_name")


    args = parser.parse_args()

    pipeline_name: str = args.pipeline_name
    fetched_pipe = pipeline_registry.get(pipeline_name, None)
    if fetched_pipe is None:
        raise KeyError(
            f"Requested pipeline does not exist: {pipeline_name}. "
            + available_pipelines_str
        )

    fetched_pipe.run()
