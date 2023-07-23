import logging
from typing import Any, Mapping

import guidance
import polars as pl

from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.utils.load_zero_shot_model import load_llm

logger = logging.getLogger(__name__)


def zero_shot_label(enriched_train_data: pl.DataFrame):
    llm = load_llm()
    program = guidance(
        """A certain student was given the following assignment.

Summarize the following text:
{{prompt_title}}
{{prompt_text}}

To which he wrote:
{{text}}

Please help me rating the 'wording' of the text that the student wrote, from a scale of -2 to 4.2

wording: {{gen 'first_digit' pattern='[0-9]' stop='.'}}.{{gen 'second_digit' pattern='[0-9]' stop='.'}}
"""
    )

    logger.info("Starting zero shot...")

    def apply_label(row):
        student_id = row[0]
        prompt_id = row[1]
        text = row[2]
        content = row[3]
        wording = row[4]
        prompt_question = row[5]
        prompt_title = row[6]
        prompt_text = row[7]

        output = program(
            prompt_title=prompt_title,
            prompt_text=prompt_text,
            text=text,
        )
        answer = f'{output["first_digit"]}.{output["second_digit"]}'
        return float(answer)

    labeled = enriched_train_data.apply(apply_label)
    enriched_train_data = enriched_train_data.with_columns(
        labeled.select("apply").to_series().alias("llama_zero_shot_wording")
    )
    return enriched_train_data


class AddZeroShotLabelTrainTask(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        enriched_train_data = context["enriched_train_data"]
        enriched_train_data = zero_shot_label(enriched_train_data)
        return {"enriched_train_data": enriched_train_data}
