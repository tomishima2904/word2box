import optuna
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

import time
from pathlib import Path
import logzero
import joblib
import os
import sys

from pl_model_optuna import LitModel
from language_modeling_with_boxes.datasets.utils import get_vocab
from language_modeling_with_boxes.datasets.utils import get_train_iter


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import file_handlers as fh


date_time = fh.get_8char_datetime()
SAVE_DIR = f"tmp/study_{date_time}"

CONFIG = {
    "batch_size": 8192,
    "box_type": "BoxTensor",
    "data_device": "cuda",
    "dataset": "example",
    "eval_file": "./data/ja_similarity_datasets/",
    "ignore_unk": True,
    "lang": "ja",
    "log_frequency": 10,
    "loss_fn": "max_margin",
    "model_type": "Word2BoxConjunction",
    "num_epochs": 4,
    "seed": 19990429,
    "add_pad": True,
    "eos_mask": True,
    "pooling": "avg_pool",
    "alpha_dim": 32,
    "cuda": True,
    "save_model": False,
}

VOCAB = get_vocab(CONFIG["dataset"], CONFIG["eos_mask"])
VOCAB_SIZE = len(VOCAB["stoi"])


def objective(trial):
    n_gram = trial.suggest_int("n_gram", 3, 10)

    # 訓練用データセットの準備
    dataset = MyDataModule(n_gram, trial, CONFIG, VOCAB)

    # モデルの定義
    model = LitModel(n_gram, trial, CONFIG, VOCAB)

    # ロガーの作成
    trial_number = trial.number
    logger = CSVLogger(SAVE_DIR, name=f"trial_{trial_number}")

    # チェックポイントの設定
    # If `save_model` is True, you can save the best model
    if CONFIG["save_model"]:
        checkpoint_callback = ModelCheckpoint(save_top_k=1)
    else:
        checkpoint_callback = ModelCheckpoint(save_top_k=0)

    # 訓練    # データローダーの定義
    subsample_thresh = trial.suggest_float("subsample_thresh", 1e-4, 1, log=True)
    train_dataloader = get_train_iter(
        CONFIG["batch_size"],
        CONFIG["dataset"],
        CONFIG["model_type"],
        n_gram,
        subsample_thresh,
        CONFIG["data_device"],
        CONFIG["add_pad"],
        CONFIG["eos_mask"],
        CONFIG["ignore_unk"],
        VOCAB,
    )


    if CONFIG["cuda"]:
        # polarisはGPU2枚なので、devices=2
        trainer = Trainer(max_epochs=CONFIG["num_epochs"],
                          devices=-1,
                          accelerator="gpu",
                          strategy="ddp",
                          deterministic=True,
                          benchmark=True,
                          logger=logger,
                          callbacks=[checkpoint_callback])
    else:
        trainer = Trainer(max_epochs=CONFIG["num_epochs"],
                          accelerator="cpu",
                          deterministic=True,
                          logger=logger,
                          callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloader)

    # 最後のエポックのロスを取得
    last_epoch_metrics = trainer.logged_metrics[-1]
    last_epoch_loss = last_epoch_metrics["epoch_loss"]
    last_epoch_score = last_epoch_metrics["best_ws_score"]

    return last_epoch_loss, last_epoch_score


if __name__ == "__main__":

    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    logzero.logfile(f"{SAVE_DIR}/logfile.log", disableStderrLogger=True)

    study = optuna.create_study(directions=["minimize", "maximize"])
    study.optimize(objective, n_trials=50)

    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    trial_with_lowest_loss = min(study.best_trials, key=lambda t: t.values[0])
    logzero.logger.info(f"Trial with highest accuracy: ")
    logzero.logger.info(f"number: {trial_with_lowest_loss.number}")
    logzero.logger.info(f"params: {trial_with_lowest_loss.params}")
    logzero.logger.info(f"values: {trial_with_lowest_loss.values}")

    joblib.dump(study, f"{SAVE_DIR}/study.pkl")
