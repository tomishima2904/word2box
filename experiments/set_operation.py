import torch
import torchtext
from torch import Tensor, LongTensor
from torch.utils.data import DataLoader
import sys, os
from typing import Union, List, Dict


# モデルのリロードに必要なモジュールをインポートする
# Import modules to reload trained model
from language_modeling_with_boxes.models import Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss
from language_modeling_with_boxes.datasets.utils import get_iter_on_device
from language_modeling_with_boxes.box.box_wrapper import BoxTensor


# 訓練済みのモデルで集合演算を行う
# Set operation using trained model
def intersection_words(
    words: LongTensor,
    model: Union[Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss],
) -> BoxTensor:
    """複数の刺激語のトークンの共通部分のboxを求める

    Args:
        words (LongTensor): 刺激語のidのテンソル. ex. [56, 9, 100]
        model (Union[Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss]): 学習済みモデル

    Returns:
        BoxTensor: wordsの共通部分のbox
    """

    with torch.no_grad():

        # 刺激語を埋め込み表現に変換
        # Embedding words
        word_boxes = model.embeddings_word(words)  # [num_stimuli, 2, embedding_dim]

        # 共通部分の X- と X+ を算出 [embedding_dim]
        # Make an intersection box from word_boxes
        intersection_z = torch.max(word_boxes.z, dim=-2).values
        intersection_Z = torch.min(word_boxes.Z, dim=-2).values
        intersection_box = BoxTensor.from_zZ(intersection_z, intersection_Z)

        return intersection_box


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
