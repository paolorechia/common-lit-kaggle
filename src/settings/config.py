import pathlib

DATA_ROOT_DIR = pathlib.Path("/home/paolo/kaggle/common-lit-kaggle/data")

DATA_INPUT_DIR = pathlib.Path(DATA_ROOT_DIR / "input")
DATA_INTERMEDIATE_DIR = pathlib.Path(DATA_ROOT_DIR / "intermediate")


INPUT_CSV = "summaries_train.csv"
INPUT_CSV_FULL_PATH = pathlib.Path(DATA_INPUT_DIR, INPUT_CSV)


MODELS_ROOT_DIR = pathlib.Path(DATA_ROOT_DIR / "models")
