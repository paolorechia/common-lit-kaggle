import logging
from typing import Any, Mapping

import nlpaug.augmenter.word as naw
import polars as pl
from tqdm import tqdm

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tables import AugmentedWord2VecTrainTable

from .bucket import BucketResult

logger = logging.getLogger(__name__)


class AugmentWord2VecTrainDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        config = Config.get()

        train_data: pl.DataFrame = context["train_data"]
        bucket_result: BucketResult = context["bucket_train_data"]

        aug = naw.WordEmbsAug(
            model_type="word2vec",
            model_path=config.models_root_dir
            / "word2vec"
            / "GoogleNews-vectors-negative300.bin",
            action="substitute",
            aug_p=0.1,
        )

        def augment_text(text):
            augmented = aug.augment(text, n=1)[0]
            return augmented

        # Process category
        for attr in ["content", "wording"]:
            samples: list[pl.DataFrame] = []
            if attr == "content":
                histogram = bucket_result.content_histogram
                max_count = bucket_result.content_max_label_size
            else:
                continue
                # histogram = bucket_result.wording_histogram
                # min_count = bucket_result.wording_min_label_size

            for category in tqdm(histogram.select("category").to_numpy()):
                tuple_string = category[0]
                tuple_string = tuple_string.replace("(", "").replace("]", "")
                tuple_ = tuple_string.split(",")
                min_ = float(tuple_[0])
                max_ = float(tuple_[1])
                bucket = train_data.filter(pl.col(attr) > min_).filter(
                    pl.col(attr) < max_
                )
                data_points_to_generate = max_count - len(bucket)
                logger.info(
                    "Will generate '%s' for bucket '%s' of label '%s'",
                    data_points_to_generate,
                    tuple_string,
                    attr,
                )
                to_augment = bucket.sample(
                    data_points_to_generate, with_replacement=True
                )
                to_augment.with_columns(
                    pl.col("text").apply(augment_text).alias("augmented_text")
                )

        augmented_samples = pl.concat(samples)
        table_io.write_table(augmented_samples, AugmentedWord2VecTrainTable())

        # Replace train data in pipeline with undersampled
        return {"augmented_train": augmented_samples}
