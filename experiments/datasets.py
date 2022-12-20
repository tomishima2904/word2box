import torch
from torch import LongTensor, BoolTensor, Tensor
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Union

from pytorch_utils import TensorDataLoader

from language_modeling_with_boxes.box.box_wrapper import BoxTensor
from language_modeling_with_boxes.models import Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss


# Dataset to evaluate similarity from all vocabulary
# Vocabulary にある全ての語彙に対して類似度を計算するための評価用データセット
class TrainedAllVocabDataset(Dataset):
    def __init__(
        self,
        vocab_stoi: dict,
        model: Union[Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss]
    ):
        super().__init__()
        self.vocab_stoi = vocab_stoi
        self.vocab_ids = [v_id for v_id in self.vocab_stoi.values()]
        # special_token = [0, 1, 2, 3, 4]
        # for token in special_token: self.vocab_ids.remove(token)
        self.vocab_ids = LongTensor(self.vocab_ids)
        self.model = model


    def __getitem__(self, idx) -> LongTensor:
        with torch.no_grad():
            # Embedding all vocab using trained model
            label: LongTensor =  self.vocab_ids[idx]
            box: BoxTensor = self.model.embeddings_word(label)

            # Convert TensorBox to LongTensor size of which is [len(vocab_stoi), 2, embedding_dim]
            box = torch.stack([box.z, box.Z], dim=-2)

            return box, label


    def __len__(self) -> int:
        return len(self.vocab_ids)
