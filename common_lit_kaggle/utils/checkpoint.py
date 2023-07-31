from pathlib import Path

from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.utils.mlflow_wrapper import mlflow


def get_checkpoint_path(passed_model_name=None, existing_run_id=None, epoch=None):
    config = Config.get()
    if passed_model_name:
        model_name = passed_model_name

    else:
        model_name = config.bart_model

    model_name = model_name.replace("/", "-")

    if existing_run_id:
        run_id = existing_run_id
    else:
        run = mlflow.active_run()
        run_id = run.info.run_id

    if epoch:
        checkpoint_path = Path(
            config.checkpoints_dir / f"trained_{model_name}_{run_id}_{epoch}"
        )
    else:
        checkpoint_path = Path(
            config.checkpoints_dir / f"trained_{model_name}_{run_id}"
        )

    return checkpoint_path
