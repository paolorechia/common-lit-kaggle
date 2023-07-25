from typing import Any, Mapping

import polars as pl

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config


class PrepareTensorDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        config = Config.get()

        return {}
