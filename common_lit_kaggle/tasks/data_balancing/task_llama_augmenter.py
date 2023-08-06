import random
from typing import Any, Mapping

import polars as pl
from tqdm import tqdm

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tables import AugmentedLlamaTrainTable
from common_lit_kaggle.tasks.data_split.data_blocks import data_blocks_generator

# pylint: disable=invalid-name
imported_auto_gptq = False
try:
    from auto_gptq import AutoGPTQForCausalLM
    from transformers import AutoTokenizer

    imported_auto_gptq = True

except ImportError as excp:
    print("Import error!", excp)


class LlamaAugmenterTask(Task):
    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)

    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        assert imported_auto_gptq, "Cannot run llama augmenter task without autogptq!"

        train_data: pl.DataFrame = context["train_unified_text_data"]

        config = Config.get()

        tokenizer, model = self._load_llama(config.llama_path)
        number_of_few_shot_examples = 4

        new_data_points = []
        # Consume generator so we know how many blocks we have for tqdm

        old_augmented = None
        try:
            old_augmented = table_io.read_table(AugmentedLlamaTrainTable())
        # pylint: disable=broad-exception-caught
        except Exception:
            pass

        blocks = list(data_blocks_generator(train_data))
        for data_block in tqdm(blocks):
            sample = data_block.sample(
                max(number_of_few_shot_examples, len(data_block))
            )
            sample_content_mean = sample["content"].mean()
            sample_wording_mean = sample["wording"].mean()
            prompt = self._samples_to_prompt(sample)

            input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
            output_ids = model.generate(**input_ids, max_new_tokens=1024)
            output = tokenizer.decode(output_ids[0])

            parsed_output = self._output_parser(prompt, output)
            assert len(parsed_output) > 0

            new_data_points.append(
                {
                    "student_id": f"SYNTHETIC_DATA_{random.randint(0, 999999999)}",
                    "prompt_id": f"SYNTHETIC_DATA_{random.randint(0, 999999999)}",
                    "content": sample_content_mean,
                    "wording": sample_wording_mean,
                    "unified_text": parsed_output,
                    "unified_labels": "NOT_PROVIDED",
                }
            )
            new_data_points = pl.DataFrame(new_data_points)  # type: ignore

            if old_augmented:
                augmented = pl.concat([old_augmented, new_data_points])  # type: ignore
            else:
                augmented = new_data_points

            table_io.write_table(augmented, AugmentedLlamaTrainTable())  # type: ignore

        return {"llama_augmented": augmented}

    def _output_parser(self, _: str, output: str):
        return output

    def _samples_to_prompt(self, sample: pl.DataFrame) -> str:
        # Available columns to use in prompt
        # student_id = pl.Utf8
        # prompt_id = pl.Utf8
        # content = pl.Float64
        # wording = pl.Float64
        # unified_text = pl.Utf8
        # unified_labels = pl.Utf8

        unified_text_list = sample.select("unified_text").to_numpy().tolist()
        # Flatten list
        unified_text_list = [text[0] for text in unified_text_list]
        print(len(unified_text_list))
        prompt = "\n".join(unified_text_list)
        return prompt

    def _load_llama(self, model_directory):
        config = Config.get()
        model = AutoGPTQForCausalLM.from_quantized(
            model_directory, device=config.device, use_safetensors=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_directory, use_fast=True)
        return tokenizer, model
