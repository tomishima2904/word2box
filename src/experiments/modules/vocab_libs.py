import torch
from torch import Tensor, LongTensor
from typing import Union, List, Dict
import json


class VocabLibs(object):
    def __init__(self, vocab_stoi_path: str) -> None:
        super().__init__()
        self.vocab_stoi: Dict = json.load(open(vocab_stoi_path, "r"))


    def words_list_to_ids_tensor(self, words: List) -> LongTensor:
        """Convert 2-dim or 1-dim words of list to IDs of LongTensor
        単語のリストをIDのテンソルに変換する関数

        Args:
            words (List): 2-dim or 1-dim list of words

        Returns:
            LongTensor: 2-dim or 1-dim LongTensor of IDs
        """
        # if `words` is 2-dim (list of list of words)
        if type(words[0]) == list:
            word_ids = []
            for word in words:
                word_ids.append(list(map(lambda s: self.stoi_considering_unk(s), word)))

        # if `words` is 1-dim (list of words)
        else:
            word_ids = list(map(lambda s: self._stoi_unk(s), words))

        return LongTensor(word_ids)


    def stoi_considering_unk(self, word: str) -> int:
        # Convert `word` to id. If `word` not in vocabulary, convert to <unk>'s id.
        try:
            return self.vocab_stoi[word]
        except:
            return self.vocab_stoi['<unk>']


    def get_vocab_list(self) -> List:
        return [k for k in self.vocab_stoi.keys()]


    def get_vocab_ids_list(self) -> List:
        return range(len(self.vocab_stoi))


    def get_vocab_itos(self) -> List:
        return [k for k, v in sorted(self.vocab_stoi.items(), key=lambda item: item[1])]


    def get_vocab_size(self) -> int:
        return len(self.vocab_stoi)
