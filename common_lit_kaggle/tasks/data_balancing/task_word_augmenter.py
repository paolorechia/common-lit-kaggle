import logging
from typing import Any, Mapping

import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
import polars as pl
from tqdm import tqdm

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tables import (
    AugmentedBertTrainTable,
    AugmentedGPT2TrainTable,
    AugmentedPPDBTrainTable,
    AugmentedT5TrainTable,
    AugmentedWmt19TrainTable,
    AugmentedWord2VecTrainTable,
)

from .bucket import BucketResult

logger = logging.getLogger(__name__)


class Augmenter:
    # pylint: disable=unused-argument
    def augment(self, text: str, **kwargs) -> list[str]:
        return [text]


def apply_augmenter(context, augmenter: Augmenter) -> pl.DataFrame:
    train_data: pl.DataFrame = context["train_data"]
    bucket_result: BucketResult = context["bucket_train_data"]

    progress_bar = tqdm(total=1)

    def augment_text(text):
        augmented = augmenter.augment(text, n=1)[0]
        progress_bar.update(1)
        return augmented

    samples: list[pl.DataFrame] = []

    # Process category
    for attr in ["content", "wording"]:
        if attr == "content":
            histogram = bucket_result.content_histogram
            max_count = bucket_result.content_max_label_size
        else:
            continue
            # histogram = bucket_result.wording_histogram
            # min_count = bucket_result.wording_min_label_size

        for category in histogram.select("category").to_numpy():
            tuple_string = category[0]
            tuple_string = tuple_string.replace("(", "").replace("]", "")
            tuple_ = tuple_string.split(",")
            min_ = float(tuple_[0])
            max_ = float(tuple_[1])
            bucket = train_data.filter(pl.col(attr) > min_).filter(pl.col(attr) < max_)
            data_points_to_generate = max_count - len(bucket)
            if data_points_to_generate > 0:
                logger.info(
                    "Will generate '%s' for bucket '%s' of label '%s'",
                    data_points_to_generate,
                    tuple_string,
                    attr,
                )
                progress_bar = tqdm(total=data_points_to_generate)
                to_augment = bucket.sample(
                    data_points_to_generate, with_replacement=True
                )
                augmented = to_augment.with_columns(
                    pl.col("text").apply(augment_text).alias("augmented_text")
                )
                samples.append(augmented)

    augmented_samples = pl.concat(samples)
    return augmented_samples


# pylint: disable=broad-exception-caught


class AugmentWord2VecTrainDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        config = Config.get()

        try:
            table = table_io.read_table(AugmentedWord2VecTrainTable())
            return {"augmented_train": table}
        except Exception:
            logger.warning("Table not found, running augmentation")

        aug = naw.WordEmbsAug(
            model_type="word2vec",
            model_path=config.models_root_dir
            / "word2vec"
            / "GoogleNews-vectors-negative300.bin",
            action="substitute",
            aug_p=0.1,
        )
        augmented_samples = apply_augmenter(context, aug)
        table_io.write_table(augmented_samples, AugmentedWord2VecTrainTable())

        # Replace train data in pipeline with undersampled
        return {"augmented_train": augmented_samples}


class AugmentT5TrainDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        config = Config.get()
        try:
            table = table_io.read_table(AugmentedT5TrainTable())
            return {"augmented_train": table}
        except Exception:
            logger.warning("Table not found, running augmentation")

        aug = nas.AbstSummAug(
            model_path=config.models_root_dir / "t5-base", device="gpu"
        )
        augmented_samples = apply_augmenter(context, aug)
        table_io.write_table(augmented_samples, AugmentedT5TrainTable())

        # Replace train data in pipeline with undersampled
        return {"augmented_train": augmented_samples}


class AugmentBertTrainDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        config = Config.get()
        try:
            table = table_io.read_table(AugmentedBertTrainTable())
            return {"augmented_train": table}
        except Exception:
            logger.warning("Table not found, running augmentation")

        aug = naw.ContextualWordEmbsAug(
            model_path=config.models_root_dir / "bert-base-uncased",
            action="substitute",
            device="gpu",
        )

        augmented_samples = apply_augmenter(context, aug)
        table_io.write_table(augmented_samples, AugmentedBertTrainTable())

        # Replace train data in pipeline with undersampled
        return {"augmented_train": augmented_samples}


class AugmentPPDBTrainDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        config = Config.get()
        try:
            table = table_io.read_table(AugmentedPPDBTrainTable())
            return {"augmented_train": table}
        except Exception:
            logger.warning("Table not found, running augmentation")

        aug = naw.SynonymAug(
            aug_src="ppdb",
            model_path=config.models_root_dir / "ppdb" / "ppdb-2.0-s-all",
        )
        augmented_samples = apply_augmenter(context, aug)
        table_io.write_table(augmented_samples, AugmentedPPDBTrainTable())

        # Replace train data in pipeline with undersampled
        return {"augmented_train": augmented_samples}


class AugmentGPT2VecTrainDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        config = Config.get()
        try:
            table = table_io.read_table(AugmentedGPT2TrainTable())
            return {"augmented_train": table}
        except Exception:
            logger.warning("Table not found, running augmentation")

        aug = nas.ContextualWordEmbsForSentenceAug(
            model_path=config.models_root_dir / "gpt2",
            device="gpu",
        )
        augmented_samples = apply_augmenter(context, aug)
        table_io.write_table(augmented_samples, AugmentedGPT2TrainTable())

        # Replace train data in pipeline with undersampled
        return {"augmented_train": augmented_samples}


class AugmentWMT19TrainDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        config = Config.get()
        try:
            table = table_io.read_table(AugmentedWmt19TrainTable())
            return {"augmented_train": table}
        except Exception:
            logger.warning("Table not found, running augmentation")

        aug = naw.BackTranslationAug(
            from_model_name=config.models_root_dir / "facebook/wmt19-en-de",
            to_model_name=config.models_root_dir / "facebook/wmt19-de-en",
            device="gpu",
        )

        augmented_samples = apply_augmenter(context, aug)
        table_io.write_table(augmented_samples, AugmentedWmt19TrainTable())

        # Replace train data in pipeline with undersampled
        return {"augmented_train": augmented_samples}
