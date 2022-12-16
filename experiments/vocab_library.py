import torch
from torch import Tensor, LongTensor
from typing import Union, List, Dict


class VocabLibrary(object):
    def __init__(self, vocab_stoi: Dict) -> None:
        super().__init__()
        self.vocab_stoi: Dict = vocab_stoi
        self.vocab_itos: Dict = [k for k, v in sorted(self.vocab_stoi.items(), key=lambda item: item[1])]
        self.vocab: List = [k for k in self.vocab_stoi.keys()]
        self.vacab_ids: List = [v for v in self.vocab_stoi.values()]


    def stoi_converter(self, words: List) -> LongTensor:
        """Convert 2-dim or 1-dim str list to IDs LongTensor
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
                word_ids.append(list(map(lambda s: self._stoi_unk(s), word)))

        # if `words` is 1-dim (list of words)
        else:
            word_ids = list(map(lambda s: self._stoi_unk(s), words))

        return LongTensor(word_ids)


    def _stoi_unk(self, word: str) -> int:
        # Convert `word` to id. If `word` not in vocabulary, convert to <unk>'s id.
        if word in self.vocab:
            return self.vocab_stoi[word]
        else:
            return self.vocab_stoi['<unk>']
