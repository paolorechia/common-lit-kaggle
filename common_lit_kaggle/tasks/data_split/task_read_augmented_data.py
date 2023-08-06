from typing import Any, Mapping

import polars as pl

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.tables import (
    AugmentedBertTrainTable,
    AugmentedGPT2TrainTable,
    AugmentedLlamaTrainTable,
    AugmentedPPDBTrainTable,
    AugmentedT5TrainTable,
    AugmentedWmt19TrainTable,
    AugmentedWord2VecTrainTable,
)


class ReadLlamaUnifiedTrainDataTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = context["train_unified_text_data"]
        llama_data = table_io.read_table(AugmentedLlamaTrainTable())
        print(len(input_data))
        print(len(llama_data))
        concated = pl.concat(
            [
                input_data.drop("text")
                .drop("prompt_question")
                .drop("prompt_title")
                .drop("prompt_text"),
                llama_data.drop("unified_labels"),
            ]
        )
        print(len(concated))
        # Merge augmetned with original train data
        return {"train_unified_text_data": concated}


class ReadWord2VecTrainTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = table_io.read_table(AugmentedWord2VecTrainTable())
        return {"word2vec_augmented_train_data": input_data}


class ReadGPT2TrainTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = table_io.read_table(AugmentedGPT2TrainTable())
        return {"gpt2_augmented_train_data": input_data}


class ReadWMT19TrainTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = table_io.read_table(AugmentedWmt19TrainTable())
        return {"wmt19_augmented_train_data": input_data}


class ReadT5TrainTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = table_io.read_table(AugmentedT5TrainTable())
        return {"t5_augmented_train_data": input_data}


class ReadPPDBTrainTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = table_io.read_table(AugmentedPPDBTrainTable())
        return {"ppdb_augmented_train_data": input_data}


class ReadBertTrainTask(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data = table_io.read_table(AugmentedBertTrainTable())
        return {"bert_augmented_train_data": input_data}


class MergeAugmentedSourcesTask(Task):
    def __init__(self, data_sources: list[str], name: str = "") -> None:
        super().__init__(name)
        self.data_sources = data_sources

    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        input_data: pl.DataFrame = context["train_data"]

        for data_source in self.data_sources:
            augmented_data: pl.DataFrame = context[data_source]
            augmented_data = augmented_data.drop("text")
            augmented_data = augmented_data.rename({"augmented_text": "text"})

        return {
            "train_data": pl.concat(
                [
                    input_data.select(sorted(input_data.columns)),
                    augmented_data.select(sorted(augmented_data.columns)),
                ]
            )
        }
