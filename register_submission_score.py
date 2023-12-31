import argparse
from common_lit_kaggle.utils.mlflow_wrapper import mlflow


def submit_score(run_id, score):
    with mlflow.start_run(run_id=run_id) as _:
        mlflow.log_metric("submission_score", score, 1)
        mlflow.log_metric("submission_score", score, 7)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "register_submission_score"
    )
    parser.add_argument("run_id")
    parser.add_argument("score")

    args = parser.parse_args()

    submit_score(args.run_id, args.score)