"""This configuration module could use some refactor to reduce duplication too"""
import os
import pathlib

# pylint: disable=too-many-statements, too-many-branches


class Config:
    _config = None

    @classmethod
    def get(
        cls,
        root_dir=os.getenv(
            "KAGGLE_DATA_DIR", "/home/paolo/kaggle/common-lit-kaggle/data"
        ),
        model_config_root_dir=os.getenv(
            "MODEL_CONFIG_DIR", "/home/paolo/kaggle/common-lit-kaggle/model_config"
        ),
        input_dir=None,
        output_dir=None,
        sentence_transformer="sentence-transformers/all-MiniLM-L6-v2",
        zero_shot_model="facebook/bart-large-mnli",
        train_prompts=None,
        test_prompts=None,
        eval_prompts=None,
        used_features=None,
        # zero_shot_model="/home/paolo/kaggle/common-lit-kaggle/data/models/Llama-2-7b-chat-hf",
        model="facebook/bart-base",
        tokenizer="facebook/bart-base",
        # model="facebook/bart-large",
        # tokenizer="facebook/bart-large",
        # model="facebook/bart-large-cnn",
        # tokenizer="facebook/bart-large-cnn",
        # model="microsoft/deberta-v3-xsmall",
        # tokenizer="microsoft/deberta-v3-xsmall",
        # model="google/pegasus-x-base",
        # tokenizer="google/pegasus-x-base",
        run_with_small_sample=False,
        num_train_epochs=10,
        batch_size=2,
        # model="/home/paolo/kaggle/common-lit-kaggle/data/models/falcon-rw-1b",
        # tokenizer="tiiuae/falcon-rw-1b",
        save_checkpoints=True,
        learning_rate=0.0001,
        regression_dropout=0.1,
        gradient_accumulation_steps=1,
    ):
        if cls._config is None:
            Config._config = cls(
                root_dir,
                model_config_root_dir,
                input_dir,
                output_dir,
                sentence_transformer,
                zero_shot_model,
                train_prompts,
                test_prompts,
                eval_prompts,
                used_features,
                tokenizer,
                model,
                run_with_small_sample,
                num_train_epochs,
                batch_size,
                save_checkpoints,
                learning_rate,
                regression_dropout,
                gradient_accumulation_steps,
            )

        return Config._config

    def __init__(
        self,
        root_dir,
        model_config_root_dir,
        input_dir,
        output_dir,
        sentence_transformer,
        zero_shot_model,
        train_prompts,
        test_prompts,
        eval_prompts,
        used_features,
        tokenizer,
        model,
        run_with_small_sample,
        num_train_epochs,
        batch_size,
        save_checkpoints,
        learning_rate,
        dropout,
        gradient_accumulation_steps,
    ) -> None:
        # Config parameters that end with _dir are automatically created by the 'main.py' script.
        self.data_root_dir = pathlib.Path(root_dir)
        self.model_config_root_dir = pathlib.Path(model_config_root_dir)

        if input_dir:
            self.data_input_dir = input_dir
        else:
            self.data_input_dir = pathlib.Path(self.data_root_dir / "input")

        self.data_intermediate_dir = pathlib.Path(self.data_root_dir / "intermediate")
        self.data_exploration_dir = pathlib.Path(self.data_root_dir / "exploration")
        self.data_train_dir = pathlib.Path(self.data_root_dir / "train")
        self.data_test_dir = pathlib.Path(self.data_root_dir / "test")
        self.data_eval_dir = pathlib.Path(self.data_root_dir / "eval")
        self.plots_dir = pathlib.Path(self.data_root_dir / "plots")
        self.models_root_dir = pathlib.Path(self.data_root_dir / "models")
        self.checkpoints_dir = pathlib.Path(self.data_root_dir / "checkpoints")
        self.llama_path = pathlib.Path(
            self.models_root_dir / "vicuna-13B-v1.5-16K-GPTQ"
        )

        self.quadratic_transform = False

        self.use_unified_text = False

        self.log_transform = False

        self.cost_sensitive_learning = False
        self.cost_sensitive_exponent = 1
        self.cost_sensitive_sum_operand = 2

        self.model_custom_config_dir = pathlib.Path(self.model_config_root_dir / model)

        # Undersampling settings
        self.min_count_multiplier = 3

        # Used to backtest a training with test split
        self.existing_run_id = "325a3949cc90415c810ddad14d18f680"

        self.dropout = dropout
        self.tokenizer = tokenizer

        # Step Linear Rate Scheduler config
        self.step_lr_step_size = 1  # Triggered every epoch
        self.step_lr_gamma = 0.9  # Multiplicative factor

        # Early stop
        self.early_stop_patience = 3
        self.early_stop_min_delta = 0.1

        self.tokenizer = tokenizer
        # Bart Base
        self.batch_size = batch_size
        self.model = model

        if "bart-base" in model:
            self.string_truncation_length = (
                1500  # value set on trial and error, until it stopped issuing warnings
            )
            self.model_context_length = 768
        elif "bart-large" in model:
            # Large bart
            self.model_context_length = 1024
            self.string_truncation_length = (
                2700  # value set on trial and error, until it stopped issuing warnings
            )
        elif "falcon" in model:
            # Large bart
            self.model_context_length = 1024
            self.string_truncation_length = (
                2700  # value set on trial and error, until it stopped issuing warnings
            )
        elif "deberta" in model:
            self.model_context_length = 512
            self.string_truncation_length = 1350

        elif "pegasus" in model:
            self.model_context_length = 2048
            self.string_truncation_length = 4000

        else:
            raise ValueError(
                f"Unknown model: '{model}'. Could not set preprocessing parameters."
            )

        self.using_stack = True
        self.number_of_models_in_stack = 2
        if self.using_stack:
            self.model_context_length *= self.number_of_models_in_stack
            self.string_truncation_length *= self.number_of_models_in_stack

        # Shared bart parameters
        self.save_checkpoints = save_checkpoints
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.num_of_labels = 2
        self.run_with_small_sample = run_with_small_sample
        self.small_sample_size = 10
        self.gradient_accumulation_steps = gradient_accumulation_steps

        if output_dir:
            self.data_output_dir = output_dir
        else:
            self.data_output_dir = pathlib.Path(self.data_root_dir / "output")

        # Only used for basic_ml pipelines
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
        self.train_prompts = ["3b9047", "39c16e", "ebad26"]
        self.eval_prompts = ["814d6b"]
        self.test_prompts = [
            "814d6b",
        ]

        if train_prompts is not None:
            self.train_prompts = train_prompts

        if test_prompts is not None:
            self.test_prompts = test_prompts

        if eval_prompts is not None:
            self.eval_prompts = eval_prompts

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
