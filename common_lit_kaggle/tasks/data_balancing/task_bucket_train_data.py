from typing import Any, Mapping

from common_lit_kaggle.framework.task import Task

from .bucket import bucket_data


class BucketTrainDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = context["train_data"]
        bucket_train_data = bucket_data(input_data)
        return {"bucket_train_data": bucket_train_data}
