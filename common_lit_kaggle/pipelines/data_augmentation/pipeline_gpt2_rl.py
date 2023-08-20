import json

from common_lit_kaggle.framework import Pipeline
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tasks import reinforcement_learning


class AugmentGPT2RL(Pipeline):
    def __init__(self) -> None:
        config = Config.get()

        super().__init__(
            "gpt2_rl_gen",
            [reinforcement_learning.GPT2Generation()],
        )
