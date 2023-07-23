from typing import Any, Mapping

from features import add_basic_features
from framework.task import Task


class AddBasicFeaturesTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        train_data = context["train_data"]
        enriched_train_data = add_basic_features(train_data)
        return {"enriched_train_data": enriched_train_data}
