import torch
from torch import Tensor, LongTensor
from torch.utils.data import DataLoader
import torch.nn as nn

import time
from pathlib import Path
import logzero
import csv
import numpy as np
from tensorboardX import SummaryWriter
import pandas as pd
import argparse
import random
import joblib
import os
import sys

import optuna

from language_modeling_with_boxes.datasets.utils import get_train_iter, get_vocab
from language_modeling_with_boxes.models import \
    Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss
from language_modeling_with_boxes.train.Trainer import TrainerWordSimilarity
from language_modeling_with_boxes.train.negative_sampling import \
    RandomNegativeCBOW, RandomNegativeSkipGram
from utils import file_handlers as fh

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


CONFIG = {
    "batch_size": 8192,
    "box_type": "CenterBoxTensor",
    "data_device": "cuda:1",
    "dataset": "jawiki",
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

W2V_DIRS = {
    50: "word2vec_ja_d50",
    100: "word2vec_ja_d100",
    200: "word2vec_ja_d200",
    300: "word2vec_ja_d300",
}


torch.manual_seed(CONFIG["seed"])
random.seed(CONFIG["seed"])


# 既存関数を使って訓練用のデータローダーを作成
def get_train_dataloader(
    n_gram,
    trial,
    config,
    vocab,
):
    subsample_thresh = trial.suggest_float(
        "subsample_thresh", 1e-4, 1, log=True)

    train_iter = get_train_iter(
        config["batch_size"],
        config["dataset"],
        config["model_type"],
        n_gram,
        subsample_thresh,
        config["data_device"],
        config["add_pad"],
        config["eos_mask"],
        config["ignore_unk"],
        vocab,
    )
    return train_iter


def define_model(
    n_gram,
    trial,
    config,
    vocab_size,
):
    # embedding_dim, int_temp, vol_temp をoptunaで探索
    embedding_dim = trial.suggest_categorical("embedding_dim", [50, 100])
    w2v_dir = f"results/{W2V_DIRS[embedding_dim]}"
    intersection_temp = trial.suggest_categorical(
        "intersection_temp", [0.1, 0.5, 1, 2, 4, 8])
    volume_temp = trial.suggest_float("volume_temp", 1e-5, 1e2, log=True)
    offset_temp = trial.suggest_float("offset_temp", 0.1, 1.0)

    if config["model_type"] == "Word2Box":
        model = Word2Box(vocab_size=vocab_size,
                         embedding_dim=embedding_dim,
                         batch_size=config["batch_size"],
                         n_gram=n_gram,
                         intersection_temp=intersection_temp,
                         volume_temp=volume_temp,
                         box_type=config["box_type"],
                         pooling=config["pooling"],
                         w2v_dir=w2v_dir,
                         offset_temp=offset_temp)

    elif config["model_type"] == "Word2BoxConjunction":
        model = Word2BoxConjunction(vocab_size=vocab_size,
                                    embedding_dim=embedding_dim,
                                    batch_size=config["batch_size"],
                                    n_gram=n_gram,
                                    volume_temp=volume_temp,
                                    box_type=config["box_type"],
                                    w2v_dir=w2v_dir,
                                    offset_temp=offset_temp)

    else:
        raise ValueError(
            "Model type is not valid. Please enter a valid model type")

    if "cuda" in config["data_device"] and torch.cuda.is_available():
        model.to(config["data_device"])

    return model


class TrainerWordSimilarity4Optuna(TrainerWordSimilarity):
    def __init__(
        self,
        train_iter,
        val_iter,
        vocab,
        trial,
        n_gram=4,
        loss_fn="max_margin",
        model_mode="CBOW",
        lang="en",
        log_frequency=1000,
        similarity_datasets_dir=None,
        subsampling_prob=None,
        device="cpu"
    ):
        super(TrainerWordSimilarity4Optuna, self).__init__(
            train_iter=train_iter,
            val_iter=val_iter,
            vocab=vocab,
            n_gram=n_gram,
            loss_fn=loss_fn,
            model_mode=model_mode,
            lang=lang,
            log_frequency=log_frequency,
            similarity_datasets_dir=similarity_datasets_dir,
            subsampling_prob=subsampling_prob,
            device=device,
        )
        self.lr = trial.suggest_float("lr", 1e-10, 1e-1, log=True)
        self.negative_samples = trial.suggest_categorical(
            "negative_samples", [1, 2, 4, 8])
        self.margin = trial.suggest_float("margin", 1, 10)

        # If subsampling has been done earlier then word count must have been changed
        # This is an expected word count based on the subsampling prob parameters.
        if subsampling_prob != None:
            self.sampling = (
                torch.min(torch.tensor(1.0).to(self.device),
                          1 - subsampling_prob.to(self.device))
                * self.sampling
            )
        if model_mode == "CBOW":
            self.add_negatives = RandomNegativeCBOW(
                self.negative_samples, self.sampling)
        elif model_mode == "SkipGram":
            self.add_negatives = RandomNegativeSkipGram(
                self.negative_samples, self.sampling)

    def train_model(
        self, trial, model, num_epochs=100, path="./checkpoints",
        save_model=False, write_summary=False,
    ):
        # Setting up the optimizers
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params=parameters, lr=self.lr)
        metric = {}
        eval_dataset = "En-Simlex-999.Txt" if self.lang == "en" else "Jwsan-1400-Asso.Tsv"
        best_simlex_ws = -1

        # Setting Up the loss function
        for epoch in range(num_epochs):
            start_time = time.time()
            epoch_loss = []
            model.train()

            for i, batch in enumerate(self.train_iter):
                # Create negative samples for the batch
                batch = self.to(batch, self.device)
                batch = self.add_negatives(batch)

                # Start the optimization
                optimizer.zero_grad()
                score = model.forward(
                    batch["center_word"],
                    batch["context_words"],
                    batch["context_mask"],
                    train=True,
                )
                assert (
                    score.shape[-1] == self.negative_samples + 1
                )  # check the shape of the score

                # Score log_intersection_volume (un-normalised) for Word2Box
                pos_score = score[..., 0].reshape(
                    -1, 1
                )  # The first element correspond to the Positive
                neg_score = score[..., 1:].reshape(
                    -1, self.negative_samples
                )  # The rest of the elements are for negative samples
                # Calculate Loss
                loss = self.loss_fn(
                    pos_score, neg_score, margin=self.margin
                )  # Margin is not required for nll or nce
                # Handled through kwargs in loss.py
                total_loss = torch.sum(loss)
                avg_loss = torch.mean(loss)
                if torch.isnan(loss).any():
                    raise RuntimeError("Loss value is nan :(")

                # Back-propagation
                total_loss.backward()
                optimizer.step()

                for param in model.parameters():
                    if torch.isinf(param).any():
                        raise RuntimeError("parameters went to infinity")
                    if torch.isnan(param).any():
                        raise RuntimeError("parameters went to nan")
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            raise RuntimeError("Gradient went to nan")

                epoch_loss.append(avg_loss.data.item())

                # Intermediate eval
                if i % int(len(self.train_iter) / self.log_frequency) == 0:
                    # Start model eval
                    model.eval()
                    ws_metric = self.model_eval(
                        model
                    )  # This ws_metric contains correlations
                    metric.update({"epoch_loss": np.mean(epoch_loss)})
                    # Update the metric for wandb login
                    metric.update(ws_metric)

                    simlex_ws = metric[eval_dataset]
                    best_simlex_ws = max(metric[eval_dataset], best_simlex_ws)
                    metric.update({"best_simlex_ws": best_simlex_ws})
                    print(
                        "Epoch {0}| Step: {3} | Loss: {1}| spearmanr: {2}".format(
                            epoch + 1, np.mean(epoch_loss), simlex_ws, i
                        )
                    )
                    logzero.logger.info(
                        f"Epoch {epoch+1}  | Step:{i}  | Loss: {np.mean(epoch_loss)}| spearmanr: {simlex_ws}")

                    model.train()

            # Logging training loss
            metric.update({"epoch_loss": np.mean(epoch_loss)})

            model.eval()
            # This ws_metric contains correlations
            ws_metric = self.model_eval(model)

            # Update the metric
            metric.update(ws_metric)

            simlex_ws = metric[eval_dataset]
            best_simlex_ws = max(simlex_ws, best_simlex_ws)
            metric.update({"best_simlex_ws": best_simlex_ws})
            loss = np.mean(epoch_loss)
            print(
                "Epoch {0} | Loss: {1}| spearmanr: {2}".format(
                    epoch + 1, loss, simlex_ws
                )
            )
            logzero.logger.info(
                f"Epoch {epoch+1} | Loss: {loss}| spearmanr: {simlex_ws}")

            # 枝刈りを行うか判断
            trial.report(loss, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return loss, simlex_ws


# Loss func
def objective(trial):
    n_gram = trial.suggest_int("n_gram", 3, 10)
    VOCAB = get_vocab(CONFIG["dataset"], CONFIG["eos_mask"])
    VOCAB_SIZE = len(VOCAB["stoi"])

    # 語彙や訓練用データローダーの準備
    train_iter = get_train_dataloader(n_gram, trial, CONFIG, VOCAB)
    val_iter = None

    model = define_model(n_gram, trial, CONFIG, VOCAB_SIZE)

    # 訓練のためのインスタンスのセットアップ
    trainer = TrainerWordSimilarity4Optuna(train_iter=train_iter,
                                           val_iter=val_iter,
                                           vocab=VOCAB,
                                           trial=trial,
                                           n_gram=n_gram,
                                           lang=CONFIG["lang"],
                                           loss_fn=CONFIG["loss_fn"],
                                           model_mode="CBOW",
                                           log_frequency=CONFIG["log_frequency"],
                                           similarity_datasets_dir=CONFIG["eval_file"],
                                           subsampling_prob=None,
                                           device=CONFIG["data_device"],
                                           )

    # 訓練
    loss, score = trainer.train_model(model=model,
                                      trial=trial,
                                      num_epochs=CONFIG["num_epochs"],
                                      path=CONFIG.get("save_dir", False),
                                      save_model=CONFIG.get("save_model", False))

    return loss  # 枝かりをやる場合、multi objective
    # return loss, score


if __name__ == "__main__":
    date_time = fh.get_8char_datetime()
    save_dir = f"results/study_{date_time}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    logzero.logfile(f"{save_dir}/logfile.log", disableStderrLogger=True)

    study = optuna.create_study(directions=["minimize"])
    study.optimize(objective, n_trials=10)

    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    trial_with_lowest_loss = min(study.best_trials, key=lambda t: t.values[0])
    logzero.logger.info(f"Trial with highest accuracy: ")
    logzero.logger.info(f"number: {trial_with_lowest_loss.number}")
    logzero.logger.info(f"params: {trial_with_lowest_loss.params}")
    logzero.logger.info(f"values: {trial_with_lowest_loss.values}")

    joblib.dump(study, f"{save_dir}/study.pkl")
