import torch
from torch import Tensor, LongTensor, BoolTensor

import time
from pathlib import Path
import logzero
import csv
import numpy as np
from tensorboardX import SummaryWriter
import pandas as pd
import os
from xopen import xopen
from scipy.stats import spearmanr
import pprint
from typing import List, Tuple, Dict, Any, Union

import lightning.pytorch as pl

from language_modeling_with_boxes.models import \
    Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss
from language_modeling_with_boxes.train.Trainer import TrainerWordSimilarity
from language_modeling_with_boxes.train.negative_sampling import \
    RandomNegativeCBOW, RandomNegativeSkipGram
from language_modeling_with_boxes.train.loss import nll, nce, max_margin


W2V_DIRS = {
    50: None,
    100: "results/230308105015",
    200: None,
    300: None,
}


class LitModel(pl.LightningModule):
    # Required #
    def __init__(
            self,
            n_gram,
            trial,
            config,
            vocab,
            model_mode="CBOW",
            subsampling_prob=None,
        ) -> None:
        super().__init__()
        self.trial = trial
        self.config = config
        self.vocab = vocab
        self.vocab_size = len(self.vocab["stoi"])

        # For optuna
        # optuna で適した値を探索するためのパラメータ群
        embedding_dim = trial.suggest_categorical("embedding_dim", [50, 100, 200, 300])
        w2v_dir: str = W2V_DIRS[embedding_dim]
        intersection_temp = trial.suggest_categorical("intersection_temp", [0.1, 0.5, 1, 2, 4, 8])
        volume_temp = trial.suggest_float("volume_temp", 1e-5, 1e2, log=True)
        offset_temp = trial.suggest_float("offset_temp", 0.1, 1.0)
        self.negative_samples = trial.suggest_categorical("negative_samples", [1, 2, 4, 8])

        self.loss_fn = self._specify_loss_fn(config["loss_fn"])
        self.similarity_datasets_dir = config["eval_file"]

        self.metrics = {}
        if self.config["lang"] == "en":
            self.eval_dataset = "En-Simlex-999.Txt"
        elif self.config["lang"] == "ja":
            self.eval_dataset = "Jwsan-1400-Asso.Tsv"
        else:
            raise ValueError("Lang type is not valid. Please enter `en` or `ja`.")
        self.best_ws_score = -1

        sorted_freqs = torch.tensor(
            [self.vocab["freqs"].get(key, 0) for key in self.vocab["itos"]]
        )
        self.sampling = torch.pow(sorted_freqs, 0.75)
        self.sampling = self.sampling / torch.sum(self.sampling)
        # If subsampling has been done earlier then word count must have been changed
        # This is an expected word count based on the subsampling prob parameters.
        if subsampling_prob != None:
            self.sampling = (
                torch.min(torch.tensor(1.0).to(self.device), 1 - subsampling_prob.to(self.device))
                * self.sampling
            )
        if model_mode == "CBOW":
            self.add_negatives = RandomNegativeCBOW(self.negative_samples, self.sampling)
        elif model_mode == "SkipGram":
            self.add_negatives = RandomNegativeSkipGram(self.negative_samples, self.sampling)

        if config["model_type"] == "Word2Box":
            self.model = Word2Box(
                vocab_size=self.vocab_size,
                embedding_dim=embedding_dim,
                batch_size=config["batch_size"],
                n_gram=n_gram,
                intersection_temp=intersection_temp,
                volume_temp=volume_temp,
                box_type=config["box_type"],
                pooling=config["pooling"],
                w2v_dir=w2v_dir,
                offset_temp=offset_temp
            )

        elif config["model_type"] == "Word2BoxConjunction":
            self.model = Word2BoxConjunction(
                vocab_size=self.vocab_size,
                embedding_dim=embedding_dim,
                batch_size=config["batch_size"],
                n_gram=n_gram,
                volume_temp=volume_temp,
                box_type=config["box_type"],
                w2v_dir=w2v_dir,
                offset_temp=offset_temp
            )

        else:
            raise ValueError("Model type is not valid. Please enter a valid model type")


    # Required #
    def forward(self, batch: Dict):
        """
        Args:
            batch["center_word"]: LongTensor. After `add_negatives`, size==[B, ns+1].
            batch["context_words"]: LongTensor. size==[B, 2*n_gram].
            batch["context_mask"]: BoolTensor. size==[B, 2*n_gram].
        """

        # Start the optimization
        score = self.model.forward(
            batch["center_word"],
            batch["context_words"],
            batch["context_mask"],
            train=True,
        )

        # Check the shape of the score
        assert score.shape[-1] == self.negative_samples + 1

        return score


    # Required #
    def configure_optimizers(self):
        lr = self.trial.suggest_float("lr", 1e-10, 1e-1, log=True)
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        return torch.optim.Adam(params=parameters, lr=lr)


    # 毎エポック、訓練が始まる前に以下の変数を初期化
    def on_train_epoch_start(self):
        self.start_time = time.time()
        self.epoch_loss = []


    # Required #
    def training_step(self, batch, batch_idx):
        # Create negative samples for the batch
        batch = self.add_negatives(batch)

        # Forward
        score = self(batch)

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

        # For logging in validation step
        avg_loss = torch.mean(loss)
        self.epoch_loss.append(avg_loss.data.item())
        if torch.isnan(loss).any():
            raise RuntimeError("Loss value is nan :(")

        total_loss = torch.sum(loss)
        return total_loss


    # This func was copied from train/Trainer.py
    def validation_step(self, batch, batch_idx):
        ws_metric = self.model_eval(self.model)
        loss = np.mean(self.epoch_loss)
        self.metrics.update({"epoch_loss": loss})
        # Update the metric for wandb login
        self.metrics.update(ws_metric)

        simlex_ws = self.metrics[self.eval_dataset]
        self.best_ws_score = max(self.metrics[self.eval_dataset], self.best_ws_score)
        self.metrics.update({"best_ws_score": self.best_ws_score})
        epoch = self.current_epoch
        print(
            "Epoch {0}| Step: {3} | Loss: {1}| spearmanr: {2}".format(
                epoch + 1, loss, simlex_ws, batch_idx
            )
        )
        logzero.logger.info(f"Epoch {epoch+1}  | Step:{batch_idx}  | Loss: {loss}| spearmanr: {simlex_ws}")


    # Epoch終了時にもログを取る
    def on_validation_epoch_end(self):
        ws_metric = self.model_eval(self.model)
        loss = np.mean(self.epoch_loss)
        self.metrics.update({"epoch_loss": loss})
        # Update the metric for wandb login
        self.metrics.update(ws_metric)

        simlex_ws = self.metrics[self.eval_dataset]
        self.best_ws_score = max(self.metrics[self.eval_dataset], self.best_ws_score)
        self.metrics.update({"best_ws_score": self.best_ws_score})
        epoch = self.current_epoch
        print(
            "Epoch {0} | Loss: {1}| spearmanr: {2}".format(
                epoch + 1, loss, simlex_ws
            )
        )
        logzero.logger.info(f"Epoch {epoch+1} | Loss: {loss}| spearmanr: {simlex_ws}")
        self.log_dict(self.metrics)


    # Not pl's func
    # See train/loss.py
    def _specify_loss_fn(self, loss_fn:str):
        criterions = {"nll": nll, "nce": nce, "max_margin": max_margin}
        return criterions[loss_fn]


    # Not pl's func
    # See model_eval() in train/Trainer.py
    def model_eval(self):
        metrics = {}
        correlation = 0.0

        # similarity_file is expected to be in the format word1\tword2\tscore
        if self.similarity_datasets_dir is not None and self.vocab is not None:
            file_list = os.listdir(self.similarity_datasets_dir)
            for file in file_list:
                with xopen(os.path.join(self.similarity_datasets_dir, file)) as f:
                    reader = csv.reader(f, delimiter="\t")
                    real_scores = []
                    predicted_scores = []
                    missing_count = 0
                    total_count = 0
                    for row in reader:
                        row[0] = row[0].lower()
                        row[1] = row[1].lower()
                        if (
                            self.vocab["stoi"].get(row[0], "<unk>") != "<unk>"
                            and self.vocab["stoi"].get(row[1], "<unk>") != "<unk>"
                        ):
                            word1 = (
                                torch.tensor(self.vocab["stoi"][row[0]], dtype=int)
                                .unsqueeze(0)
                                .to(self.device)
                            )
                            word2 = (
                                torch.tensor(self.vocab["stoi"][row[1]], dtype=int)
                                .unsqueeze(0)
                                .to(self.device)
                            )
                            score = self.model.word_similarity(word1, word2)
                            if file.title() == "Hyperlex-Dev.Txt":
                                score = self.model.conditional_similarity(word1, word2)

                            predicted_scores.append(score.item())
                            real_scores.append(float(row[2]))
                        else:
                            missing_count += 1
                        total_count += 1
                    # print(f"{file.title()} missing data point: {missing_count} out of {total_count}")
                    # Calculate spearman's rank correlation coefficient between predicted scores and real scores
                    correlation = spearmanr(predicted_scores, real_scores)[0]
                    metrics[file.title()] = correlation

        pprint.pprint(metrics, width=1)
        logzero.logger.info(metrics)

        return metrics

