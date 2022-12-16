import torch
import torchtext
from torch import Tensor, LongTensor
from torch.utils.data import DataLoader
import pickle, json
import sys, os
from tqdm import tqdm
from typing import Union, List, Dict


# モデルのリロードに必要なモジュールをインポートする
from language_modeling_with_boxes.models import Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss
from language_modeling_with_boxes.datasets.utils import get_iter_on_device

import set_operation
from utils.file_handler import *
from datasets import TrainedAllVocabDataset
from vocab_library import VocabLibrary


# 学習用データセット train.pt の中身を見る
train_tokenized = torch.load("./data/ptb/train.pt")

# itos (IDから文字列) の辞書を作成
vocab_stoi = json.load(open("./data/ptb/vocab_stoi.json", "r"))
vocab_libs = VocabLibrary(vocab_stoi)
vocab_itos = vocab_libs.vocab_itos

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

words = ['bank', 'river']  # 刺激語のリスト
word_ids = vocab_libs.stoi_converter(words)  # IDのテンソルへ変換

# 語彙のデータローダー
dataloader = DataLoader(
    dataset= TrainedAllVocabDataset(vocab_stoi, model),
    batch_size=128,
    shuffle=False
)

# 評価用データセットをロード
datasets_dir = 'data/qualitative_datasets'
dataset_name = 'intersection_2.csv'
eval_dataframe = csv_reader(f'{datasets_dir}/{dataset_name}')
eval_words_list: List = eval_dataframe.to_numpy().tolist()
eval_ids_list: LongTensor = vocab_libs.stoi_converter(eval_words_list)
assert len(eval_words_list) == len(eval_ids_list), f"cat't match the length of `words_list` {len(eval_words_list)} and `ids_list` {len(eval_ids_list)}"

# 刺激語の共通部分のboxと全ての語彙のboxとの類似度を計算
num_stimuli = eval_ids_list.size(-1)
header = []
for i in range(num_stimuli): header.append(f"stimulus_{i+1}")
for i in range(num_stimuli): header.append(f"id_{i+1}")
header.extend(["labels", "scores"])
header = tuple(header)
results = [header]
for stimuli, stim_ids in tqdm(zip(eval_words_list, eval_ids_list), total=len(eval_words_list)):
    result = []
    result.extend(stimuli)
    scores, labels = set_operation.all_words_similarity(stim_ids, dataloader, model)
    result.extend(stim_ids.numpy().tolist())
    similar_words = [vocab_itos[label] for label in (labels).to(torch.int64)]
    result.append(similar_words)
    result.append(scores.numpy().tolist())
    results.append(tuple(result))

# 結果を出力
results_dir = "results/tmp"
makedirs(results_dir)
csv_writer(path=f"{results_dir}/tmp_{dataset_name}", data=results)
