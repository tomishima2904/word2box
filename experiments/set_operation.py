import torch
import pickle, json
import sys, os


# モデルのリロードに必要なモジュールをインポートする
from language_modeling_with_boxes.models import Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss
from language_modeling_with_boxes.datasets.utils import get_iter_on_device
from language_modeling_with_boxes.__main__ import main


# 学習用データセット train.pt の中身を見る
train_tokenized = torch.load("./data/ptb/train.pt")

# itos (IDから文字列) の辞書を作成
vocab_stoi = json.load(open("./data/ptb/vocab_stoi.json", "r"))
vocab_itos = [k for k, v in sorted(vocab_stoi.items(), key=lambda item: item[1])]

# 保存してあるモデルと同じパラメータを設定する
config = {
    "batch_size": 4096,
    "box_type": "BoxTensor",
    "data_device": "gpu",
    "dataset": "ptb",
    "embedding_dim": 64,
    "eos_mask": True,
    "eval_file": "../data/similarity_datasets/",
    "int_temp": 1.9678289474987882,
    "log_frequency": 10,
    "loss_fn": "max_margin",
    "lr": 0.004204091643267762,
    "margin": 5,
    "model_type": "Word2BoxConjunction",
    "n_gram": 5,
    "negative_samples": 2,
    "num_epochs": 10,
    "subsample_thresh": 0.001,
    "vol_temp": 0.33243242379830407,
    "save_model": "",
    "add_pad": "",
    "save_dir": "results",
}

# 語彙や訓練用データを用意（モデルのインスタンス作成のため）
TEXT, train_iter, val_iter, test_iter, subsampling_prob = get_iter_on_device(
    config["batch_size"],
    config["dataset"],
    config["model_type"],
    config["n_gram"],
    config["subsample_thresh"],
    config["data_device"],
    config["add_pad"],
    config["eos_mask"],
)

# モデルのインスタンスを作成する
model = Word2BoxConjunction(
    TEXT=TEXT,
    embedding_dim=config["embedding_dim"],
    batch_size=config["batch_size"],
    n_gram=config["n_gram"],
    intersection_temp=config["int_temp"],
    volume_temp=config["vol_temp"],
    box_type=config["box_type"],
)

# 作成したインスタンスに訓練済みモデルのパラメータを読み込む
model.load_state_dict(torch.load('results/best_model.ckpt'))

# 訓練済みのモデルで集合演算を行う
