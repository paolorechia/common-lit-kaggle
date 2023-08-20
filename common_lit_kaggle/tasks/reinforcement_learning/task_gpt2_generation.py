from typing import Any, Mapping

import polars as pl
from tqdm import tqdm
from transformers import GPT2Tokenizer

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.tables import RLGPT2SyntheticData
from trlx.trlx.models.modeling_ilql import AutoModelForCausalLMWithILQLHeads


class GPT2Generation(Task):
    def run(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        model = AutoModelForCausalLMWithILQLHeads.from_pretrained(
            "./bak_gpt2_ilql_trained"
        ).to("cuda")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        prompts = [" "]
        min_length = 2048
        generated_texts = []
        samples_to_generate = 4096
        for prompt in prompts:
            print("Starting prompt: ", prompt, len(prompt))
            for _ in tqdm(range(samples_to_generate)):
                text_output = ""
                used_prompt = prompt
                while len(text_output) < min_length:
                    output = model.generate(
                        **tokenizer([used_prompt], return_tensors="pt").to("cuda"),
                        max_length=2048,
                        temperature=2.0,
                    )
                    text_output = tokenizer.decode(output[0], skip_special_tokens=True)
                    used_prompt = text_output

                print("Expanded text ", text_output)
                generated_texts.append(text_output)
        synthetic_data = pl.from_dict({"text": generated_texts})
        table_io.write_table(synthetic_data, RLGPT2SyntheticData())
        return {}
