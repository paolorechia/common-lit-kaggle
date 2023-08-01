from typing import Any, Mapping

import polars as pl

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config

from .bucket import BucketResult


class UndersampleTrainDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        train_data: pl.DataFrame = context["train_data"]
        bucket_result: BucketResult = context["bucket_train_data"]

        config = Config.get()
        # Process category
        for attr in ["content", "wording"]:
            samples = []
            if attr == "content":
                histogram = bucket_result.content_histogram
                min_count = bucket_result.content_min_label_size
            else:
                histogram = bucket_result.wording_histogram
                min_count = bucket_result.wording_min_label_size

            min_count *= config.min_count_multiplier
            for category in histogram.select("category").to_numpy():
                tuple_string = category[0]
                tuple_string = tuple_string.replace("(", "").replace("]", "")
                tuple_ = tuple_string.split(",")
                min_ = float(tuple_[0])
                max_ = float(tuple_[1])
                bucket = train_data.filter(pl.col(attr) > min_).filter(
                    pl.col(attr) < max_
                )
                sample_size = min(len(bucket), min_count)
                label_sample = bucket.sample(sample_size)
                samples.append(label_sample)

        category_undersampled_data = pl.concat(samples)
        # Replace train data in pipeline with undersampled
        return {"train_data": category_undersampled_data}
