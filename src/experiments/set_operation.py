import torch
from torch import Tensor, LongTensor
from torch.utils.data import DataLoader
import sys, os
from typing import Union, List, Dict, Tuple
from tqdm import tqdm


# モデルのリロードに必要なモジュールをインポートする
# Import modules to reload trained model
from language_modeling_with_boxes.models import Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss
from language_modeling_with_boxes.box.box_wrapper import BoxTensor


class SetOperations(object):

    def __init__(self) -> None:
        pass

    @classmethod
    # 複数の入力語に対する共通部分のboxを求める
    # Calculate an intersection box of input multiple words using trained model
    def intersect_multiple_box(
        cls,
        word_ids: LongTensor,
        model: Union[Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss],
    ) -> BoxTensor:
        """複数の入力語の共通部分のboxを求める
        box_wrapper.BoxTensor の _intersection を参考にした.
        This func refers to box_wrapper.BoxTensor._intersection

        Args:
            word_ids (LongTensor): 刺激語のidのテンソル. Input IDs of words.
            model (Union[Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss]): 学習済みモデル. Trained model.

        Returns:
            BoxTensor: word_idsの共通部分のbox. Intersection box of input words.
        """

        with torch.no_grad():

            model.eval()
            device = word_ids.device

            # 刺激語を埋め込み表現に変換
            # Embedding words
            word_boxes = model.embeddings_word(word_ids)  # [len(word_ids), 2, embedding_dim]

            gumbel_beta: float = model.intersection_temp

            if not gumbel_beta == 0.0: bayesian = True

            # 共通部分の box を算出
            # Make an intersection box from word_boxes
            if bayesian:
                t1_z = word_boxes.z[0].to(device)  # [embedding_dim]
                t1_Z = word_boxes.Z[0].to(device)

                for t2_z, t2_Z in zip(word_boxes.z[1:], word_boxes.Z[1:]):
                    try:
                        t2_z.to(device)
                        t2_Z.to(device)
                        z = gumbel_beta * torch.logaddexp(
                            t1_z / gumbel_beta, t2_z / gumbel_beta
                        )
                        t1_z = torch.max(z, torch.max(t1_z, t2_z))
                        Z = -gumbel_beta * torch.logaddexp(
                            -t1_Z / gumbel_beta, -t2_Z / gumbel_beta
                        )
                        t1_Z = torch.min(Z, torch.min(t1_Z, t2_Z))
                    except Exception as e:
                        print("Gumbel intersection is not possible")
                        breakpoint()
                intersection_z = t1_z  # [embedding_dim]
                intersection_Z = t1_Z

            else:
                intersection_z = torch.max(word_boxes.z, dim=-2).values  # [embedding_dim]
                intersection_Z = torch.min(word_boxes.Z, dim=-2).values

            intersection_z = intersection_z.unsqueeze(0).unsqueeze(0)  # [1, 1, embedding_dim]
            intersection_Z = intersection_Z.unsqueeze(0).unsqueeze(0)
            intersection_box = BoxTensor.from_zZ(intersection_z, intersection_Z)  # [1, 1, 2, embedding_dim]

            return intersection_box


def all_words_similarity(
    word_ids: LongTensor,
    dataloader: DataLoader,
    model: Union[Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss],
) -> Tuple[LongTensor, LongTensor]:
    """ Output the most `num_output` similar scores and labels to box of input words

    Args:
        words (LongTensor): 刺激語IDのリスト. List of indices of words.
        dataloader (DataLoader): Vocabulary.
        num_output (int, optional): Top `num_output` most similar words. Defaults to 100.

    Returns:
        LongTensor: Top `num_output` scores which are the most similar to intersection box of input `words`
        LongTensor: Top `num_output` labels which are the most similar to intersection box of input `words`
    """

    device = word_ids.device

    with torch.no_grad():

        intersection_box = SetOperations.intersect_multiple_box(word_ids, model)  # [1, 1, 2, embedding_dim]
        all_scores = LongTensor([])
        all_labels = LongTensor([])

        for boxes, labels in tqdm(dataloader):

            # 共通部分と語彙のBoxTensorを作成
            # Make BoxTensors
            B = len(boxes)
            vocab_z: LongTensor = boxes[..., 0, :].unsqueeze(-2).to(device)
            vocab_Z: LongTensor = boxes[..., 1, :].unsqueeze(-2).to(device)
            vocab_boxes = BoxTensor.from_zZ(vocab_z, vocab_Z)  # [B, 1, 2, embedding_dim]
            intersection_z = intersection_box.z.expand(B, -1, -1).to(device)  # [B, 1, embedding_dim]
            intersection_Z = intersection_box.Z.expand(B, -1, -1).to(device)
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

            scores = scores.squeeze(-1).to('cpu').detach()
            assert scores.size() == labels.size(), f"can't match size of `scores {scores.size()}` and `labels {labels.size()}`"
            all_scores = torch.cat([all_scores, scores])
            all_labels = torch.cat([all_labels, labels.to('cpu').detach()])

        # scores を降順に並び替え、それに伴い labels も並び替える
        # Sort scores in descending order, also sort labels along with scores
        all_scores, sorted_indices = torch.sort(all_scores, descending=True)
        all_labels = all_labels[sorted_indices]

    return all_scores, all_labels
