import pathlib


class Config:
    _config = None

    @classmethod
    def get(
        cls,
        root_dir="/home/paolo/kaggle/common-lit-kaggle/data",
        input_dir=None,
        output_dir=None,
        sentence_transformer="sentence-transformers/all-MiniLM-L6-v2",
        zero_shot_model="facebook/bart-large-mnli",
        train_prompts=None,
        test_prompts=None,
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
    ) -> None:
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

        if output_dir:
            self.data_output_dir = output_dir
        else:
            self.data_output_dir = pathlib.Path(self.data_root_dir / "output")

        self.sentence_transformer = sentence_transformer
        self.distance_metric = "euclidean"
        self.distance_stategy = "maximum"

        self.zero_shot_model = zero_shot_model

        self.available_prompts = [
            "3b9047",
            "39c16e",
            "ebad26",
            "814d6b",
        ]

        # Default configuration locally, use only half the prompts for training
        self.train_prompts = ["3b9047", "39c16e"]
        self.test_prompts = [
            "ebad26",
            "814d6b",
        ]

        if train_prompts:
            self.train_prompts = train_prompts

        if test_prompts:
            self.test_prompts = test_prompts

        assert (
            len(self.train_prompts) > 0
        ), "At least one prompt must be used for training!"
        assert len(self.train_prompts) + len(self.test_prompts) > len(
            self.available_prompts
        ), "Invalid prompt configuration!"

        self.device = "cuda:0"
