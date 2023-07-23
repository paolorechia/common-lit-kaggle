from typing import Mapping

from .pipeline import Pipeline
from .table import TableReference
from .task import Task


def build_pipeline_registry(python_module) -> Mapping[str, Pipeline]:
    """Magically finds all top-level classes from the given Python object (typically a module).

    Instantiates all classes and returns a dictionary of the classes in the format:

    {
        "pipeline_identifier": <PipelineInstance>
    }
    """

    module_attrs_names = list(dir(python_module))
    fetched_attrs = [
        getattr(python_module, pipe)
        for pipe in module_attrs_names
        if isinstance(pipe, str)
    ]
    available_pipelines = [attrs for attrs in fetched_attrs if isinstance(attrs, type)]

    pipeline_registry = {}

    for pipe_class in available_pipelines:
        pipe = pipe_class()
        pipeline_registry[pipe.name] = pipe

    return pipeline_registry
