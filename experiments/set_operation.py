import torch
import torchtext
from torch import Tensor, LongTensor, IntTensor
from torch.utils.data import DataLoader
import sys, os
from typing import Union, List, Dict, Tuple


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
        word_boxes = model.embeddings_word(words)  # [len(words), 2, embedding_dim]

        # 共通部分の X- と X+ を算出 [embedding_dim]
        # Make an intersection box from word_boxes
        intersection_z = torch.max(word_boxes.z, dim=-2).values.unsqueeze(0).unsqueeze(0)  # [1, 1, embedding_dim]
        intersection_Z = torch.min(word_boxes.Z, dim=-2).values.unsqueeze(0).unsqueeze(0)
        intersection_box = BoxTensor.from_zZ(intersection_z, intersection_Z)  # [1, 1, 2, embedding_dim]

        return intersection_box


def all_words_similarity(
    word_ids: LongTensor,
    dataloader: DataLoader,
    model: Union[Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss],
    num_output: int = 100,
) -> Tuple[LongTensor, LongTensor]:
    """ Output most similar scores and labels

    Args:
        words (LongTensor): 刺激語IDのリスト. List of indices of words.
        dataloader (DataLoader): Vocabulary.
        num_output (int, optional): Top `num_output` most similar words. Defaults to 100.

    Returns:
        LongTensor: Top `num_output` scores which are the most similar to intersection box of input `words`
        LongTensor: Top `num_output` labels which are the most similar to intersection box of input `words`
    """

    with torch.no_grad():

        intersection_box = intersection_words(word_ids, model)  # [1, 1, 2, embedding_dim]
        all_scores = Tensor([])
        all_labels = Tensor([])

        for boxes, labels in dataloader:

            # 共通部分と語彙のBoxTensorを作成
            # Make BoxTensors
            B = len(boxes)
            vocab_z: Tensor = boxes[..., 0, :].unsqueeze(-2)  # [B, 1, embedding_dim]
            vocab_Z: Tensor = boxes[..., 1, :].unsqueeze(-2)
            vocab_boxes = BoxTensor.from_zZ(vocab_z, vocab_Z)  # [B, 1, 2, embedding_dim]
            intersection_z = intersection_box.z.expand(B, -1, -1)  # [B, 1, embedding_dim]
            intersection_Z = intersection_box.Z.expand(B, -1, -1)
            repeated_intersection_box = BoxTensor.from_zZ(intersection_z, intersection_Z)  # [B, 1, 2, embedding_dim]

            # 類似度を計算
            # Calculate similarity
            if model.intersection_temp == 0.0:
                scores = vocab_boxes.intersection_log_soft_volume(
                    repeated_intersection_box, temp=model.volume_temp
                )
            else:
                scores = vocab_boxes.gumbel_intersection_log_volume(
                    repeated_intersection_box,
                    volume_temp=model.volume_temp,
                    intersection_temp=model.intersection_temp,
                )

            scores = scores.squeeze(-1)
            assert scores.size() == labels.size(), f"can't match size of `scores {scores.size()}` and `labels {labels.size()}`"
            all_scores = torch.cat([all_scores, scores])
            all_labels = torch.cat([all_labels, labels])

            # scores を降順に並び替え、それに伴い labels も並び替える
            # Sort scores and labels in descending order
            all_scores, sorted_indices = torch.sort(all_scores, descending=True)
            all_labels = all_labels[sorted_indices]

            if len(all_scores) > num_output:
                all_scores = all_scores[:num_output]
                all_labels = all_labels[:num_output]

    return all_scores, all_labels
