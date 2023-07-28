from typing import Any, Mapping

from common_lit_kaggle.features import add_basic_features
from common_lit_kaggle.framework.task import Task


class AddBasicFeaturesTestTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        train_data = context["test_data"]
        enriched_train_data = add_basic_features(train_data)
        return {"enriched_test_data": enriched_train_data}
