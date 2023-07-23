import pathlib

DATA_ROOT_DIR = pathlib.Path("/home/paolo/kaggle/common-lit-kaggle/data")

DATA_INPUT_DIR = pathlib.Path(DATA_ROOT_DIR / "input")
DATA_INTERMEDIATE_DIR = pathlib.Path(DATA_ROOT_DIR / "intermediate")
DATA_EXPLORATION_DIR = pathlib.Path(DATA_ROOT_DIR / "exploration")

DATA_TRAIN_DIR = pathlib.Path(DATA_ROOT_DIR / "train")
DATA_TEST_DIR = pathlib.Path(DATA_ROOT_DIR / "test")

PLOTS_DIR = pathlib.Path(DATA_ROOT_DIR / "plots")

MODELS_ROOT_DIR = pathlib.Path(DATA_ROOT_DIR / "models")
