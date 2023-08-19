from typing import Any, Mapping

import polars as pl

from common_lit_kaggle.framework.task import Task
from trlx import trlx
from trlx.trlx.data.default_configs import TRLConfig, default_ilql_config


class RLGPT2Task(Task):
    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        train_data: pl.DataFrame = context["train_data"]

        hparams = {}  # type: ignore
        samples = [l[0] for l in train_data.select("text").to_numpy().tolist()]
        prompts = [p[0] for p in train_data.select("prompt_text").to_numpy().tolist()]
        rewards = [
            r[0]
            for r in train_data.with_columns(
                (pl.col("content") + pl.col("wording")).alias("unified_label")
            )
            .select("unified_label")
            .to_numpy()
            .tolist()
        ]

        print("Kaggle ", samples[0])
        print("Kaggle label", rewards[0])

        config = TRLConfig.update(default_ilql_config().to_dict(), hparams)

        # micro batch size per gpu
        config.train.batch_size = 4
        # # freeze all transformer layers
        # config.model.num_layers_unfrozen = 0
        # maximum sample length, prompts or samples longer than that will be truncated
        config.train.seq_length = 1024

        # # micro batch size for sampling (specific for PPO)
        # config.method.chunk_size = 1
        # # use an additional Q-head (specific for ILQL)
        # config.method.two_qs = False

        base_model = "gpt2"
        trainer = trlx.train(
            base_model, prompts=prompts, samples=samples, rewards=rewards, config=config
        )

        trainer.save_pretrained(f"{base_model}_ilql_trained")
        return {}
