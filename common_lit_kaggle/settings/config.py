import os
import pathlib


class Config:
    _config = None

    @classmethod
    def get(
        cls,
        root_dir=os.getenv(
            "KAGGLE_DATA_DIR", "/home/paolo/kaggle/common-lit-kaggle/data"
        ),
        input_dir=None,
        output_dir=None,
        sentence_transformer="sentence-transformers/all-MiniLM-L6-v2",
        zero_shot_model="facebook/bart-large-mnli",
        train_prompts=None,
        test_prompts=None,
        used_features=None,
        # zero_shot_model="/home/paolo/kaggle/common-lit-kaggle/data/models/Llama-2-7b-chat-hf",
    ):
        if cls._config is None:
            Config._config = cls(
                root_dir,
                input_dir,
                output_dir,
                sentence_transformer,
                zero_shot_model,
                train_prompts,
                test_prompts,
                used_features,
            )

        return Config._config

    def __init__(
        self,
        root_dir,
        input_dir,
        output_dir,
        sentence_transformer,
        zero_shot_model,
        train_prompts,
        test_prompts,
        used_features,
    ) -> None:
        # Config parameters that end with _dir are automatically created by the 'main.py' script.
        self.data_root_dir = pathlib.Path(root_dir)

        if input_dir:
            self.data_input_dir = input_dir
        else:
            self.data_input_dir = pathlib.Path(self.data_root_dir / "input")

        self.data_intermediate_dir = pathlib.Path(self.data_root_dir / "intermediate")
        self.data_exploration_dir = pathlib.Path(self.data_root_dir / "exploration")
        self.data_train_dir = pathlib.Path(self.data_root_dir / "train")
        self.data_test_dir = pathlib.Path(self.data_root_dir / "test")
        self.plots_dir = pathlib.Path(self.data_root_dir / "plots")
        self.models_root_dir = pathlib.Path(self.data_root_dir / "models")

        self.bart_model = "facebook/bart-large-cnn"
        self.string_truncation_length = (
            2700  # value set on trial and error, until it stopped issuing warnings
        )
        self.model_context_length = 1024
        self.num_train_epochs = 2
        self.batch_size = 64
        self.learning_rate = 0.001

        if output_dir:
            self.data_output_dir = output_dir
        else:
            self.data_output_dir = pathlib.Path(self.data_root_dir / "output")

        self.sentence_transformer = sentence_transformer
        self.distance_metric = "euclidean"
        self.distance_stategy = "minimum"

        self.used_features = [
            "text_length",
            "word_count",
            "sentence_count",
            "unique_words",
            "word_intersection",
            "prompt_length",
            "prompt_word_count",
            "prompt_sentence_count",
            "prompt_unique_words",
        ]

        if used_features:
            self.used_features = used_features

        self.zero_shot_model = zero_shot_model

        self.available_prompts = [
            "3b9047",
            "39c16e",
            "ebad26",
            "814d6b",
        ]

        # Default configuration locally, uses only one of the prompts for training
        self.train_prompts = ["3b9047", "39c16e"]
        self.test_prompts = [
            "ebad26",
            "814d6b",
        ]

        if train_prompts is not None:
            self.train_prompts = train_prompts

        if test_prompts is not None:
            self.test_prompts = test_prompts

        assert (
            len(self.train_prompts) > 0
        ), "At least one prompt must be used for training!"
        train = len(self.train_prompts)
        test = len(self.test_prompts)
        available = len(self.available_prompts)
        assert (
            train + test <= available
        ), f"Invalid prompt configuration! {train} + {test} > {available}"

        self.device = "cuda:0"
        self.random_state = 42
