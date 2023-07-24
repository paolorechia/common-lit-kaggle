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
        # zero_shot_model="/home/paolo/kaggle/common-lit-kaggle/data/models/Llama-2-7b-chat-hf",
    ):
        if cls._config is None:
            Config._config = cls(
                root_dir, input_dir, output_dir, sentence_transformer, zero_shot_model
            )

        return Config._config

    def __init__(
        self, root_dir, input_dir, output_dir, sentence_transformer, zero_shot_model
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

        self.zero_shot_model = zero_shot_model
        self.device = "cuda:0"
