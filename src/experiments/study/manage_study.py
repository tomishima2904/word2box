import sys
import os
from dotenv import load_dotenv
import logzero

import optuna
import torch_optuna
import pl_study


# 環境変数を読み込む
# Read environment variables
load_dotenv()
db = os.getenv("MYSQL_DATABASE")
host = os.getenv("MYSQL_HOST")
user = os.getenv("MYSQL_USER")
psword = os.getenv("MYSQL_PASSWORD")


def my_create_study(study_name: str, sampler=None) -> None:
    if sampler == "random":
        optuna.create_study(
            study_name=study_name,
            storage=f"mysql+pymysql://{user}:{psword}@db:3306/{db}",
            directions=["minimize", "maximize"],  # loss, score
            sampler=optuna.samplers.RandomSampler(),
        )
    else:
        optuna.create_study(
            study_name=study_name,
            storage=f"mysql+pymysql://{user}:{psword}@db:3306/{db}",
            directions=["minimize", "maximize"],  # loss, score
        )
    print(f"Seccuessfully study `{study_name}` is created !")


def load_and_optimize_study(study_name: str, n_trials: int, objective_type="torch") -> None:
    study = optuna.load_study(
            study_name=study_name,
            storage=f"mysql+pymysql://{user}:{psword}@db:3306/{db}",
        )
    if objective_type == "torch":
        study.optimize(torch_optuna.objective, n_trials=n_trials)
    elif objective_type == "pl":
        study.optimize(pl_study.objective, n_trials=n_trials)
    else:
        raise ValueError(f"Invalid objective type {objective_type}")


if __name__ == "__main__":

    n = 1

    # ログファイルの設定
    save_dir = f"results/study_{n}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    logzero.logfile(f"{save_dir}/logfile_1.log", disableStderrLogger=True)

    # study
    study_name = f"w2b_study_{n}"
    my_create_study(study_name, sampler="random")
    load_and_optimize_study(study_name, n_trials=5, objective_type="torch")
