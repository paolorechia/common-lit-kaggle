import logging
from typing import Iterable

from .task import Task

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, name: str, tasks: Iterable[Task]) -> None:
        self.name = name
        self.tasks = tasks

    def run(self):
        for task in self.tasks:
            logger.info("Starting task: %s", task)
            task.run()
