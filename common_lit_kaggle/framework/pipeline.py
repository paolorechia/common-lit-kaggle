import logging
from typing import Iterable

from common_lit_kaggle.framework.task import Task

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, name: str, tasks: Iterable[Task]) -> None:
        self.name = name
        self.tasks = tasks

    def run(self):
        context = {}
        for task in self.tasks:
            logger.info("Starting task: %s", task)
            result = task.run(context)
            context.update(result)
