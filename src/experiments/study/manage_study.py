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


def my_create_study(study_name: str, storage: str, sampler=None) -> None:
    if sampler == "random":
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            directions=["minimize"],  # loss, score
            sampler=optuna.samplers.RandomSampler(),
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            directions=["minimize"],  # loss, score
            load_if_exists=True,
        )
    return study


def my_load_study(study_name: str, storage: str) -> optuna.study:
    study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )
    return study


def my_optimize_study(study, n_trials: int, objective_type: str="torch") -> None:
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

    # studyの保存先を定義
    study_name = f"w2b_study_{n}"
    storage = f"mysql+pymysql://{user}:{psword}@db:3306/{db}"

    # studyを作成・読み込み
    study = my_create_study(
        study_name,
        storage=storage,
        sampler="random"
    )

    # 最適化
    my_optimize_study(study, n_trials=10, objective_type="torch")
