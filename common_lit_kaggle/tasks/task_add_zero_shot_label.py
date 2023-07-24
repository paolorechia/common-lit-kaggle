import logging
from typing import Any, Mapping

import polars as pl
from transformers import pipeline

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.utils.load_zero_shot_model import load_llm

logger = logging.getLogger(__name__)


def zero_shot_label(enriched_train_data: pl.DataFrame):
    llm = load_llm()

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    logger.info("Starting zero shot...")

    def apply_label(row):
        text = row[2]
        wording = row[4]
        prompt_text = row[7]

        sequence_to_classify = f"Question:{prompt_text}\nSummary:{text}"
        candidate_labels = [
            "terrible_wording_summary",
            "bad_wording_summary",
            "ok_wording_summary",
            "good_wording_summary",
            "excellent_wording_summary",
        ]
        result = classifier(sequence_to_classify, candidate_labels)
        labels = result["labels"]
        max_score = -1
        max_label_idx = -1
        for idx, score in enumerate(result["scores"]):
            if score > max_score:
                max_score = score
                max_label_idx = idx

        label = labels[max_label_idx]
        print(label, wording)

    labeled = enriched_train_data.apply(apply_label)
    labeled.write_csv("bart_labeled.csv")
    print(labeled)
    enriched_train_data = enriched_train_data.with_columns(
        labeled.select("apply").to_series().alias("llama_zero_shot_wording")
    )
    return enriched_train_data


class AddZeroShotLabelTrainTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        enriched_train_data = context["enriched_train_data"]
        enriched_train_data = zero_shot_label(enriched_train_data)
        return {"enriched_train_data": enriched_train_data}
