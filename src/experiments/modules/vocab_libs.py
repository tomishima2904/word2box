import torch
from torch import Tensor, LongTensor
from typing import Union, List, Dict
import json
import numpy as np
import matplotlib.pyplot as plt


class VocabLibs(object):
    def __init__(self, vocab_stoi_path: str) -> None:
        super().__init__()
        print("Loading vocab ...")
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
            word_ids = list(map(lambda s: self.stoi_considering_unk(s), words))

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


class VocabLibsWithFreq(VocabLibs):
    def __init__(self, vocab_stoi_path: str, vocab_freq_path: str = None) -> None:
        super().__init__(vocab_stoi_path)
        self.vocab_freq: Dict = json.load(open(vocab_freq_path, "r"))


    # Sort freq
    def _sort_freq(self) -> List:
        freq_list = [v for v in self.vocab_freq.values()]
        freq_list.sort(reverse=True)
        return freq_list


    # Dump sorted vocab freqs (without tokens)
    def dump_sorted_freq(self, output_path) -> None:
        freq_list = self._sort_freq()
        with open(output_path, "w") as f:
            for freq in freq_list:
                print(freq, file=f)


    # Plot sorted vocab freqs
    def plot_sorted_freq(self, input_path, output_dir, x_limit=None) -> None:
        with open(input_path, "r") as f:
            sorted_freqs = np.array([int(line) for line in f])

            if x_limit != None:
                sorted_freqs = sorted_freqs[:x_limit]
                save_path = f"{output_dir}/sorted_freq_{x_limit}.png"
            else:
                save_path = f"{output_dir}/sorted_freq_all.png"

            x = np.arange(len(sorted_freqs))
            plt.yscale('log')
            plt.plot(x, sorted_freqs)
            plt.title('Sorted frequency')
            plt.xlabel('Vocab')
            plt.ylabel('Frequency')
            plt.savefig(save_path, format="png")


    # Count num of vocab freq of which is less than x
    def count_less_freq(self, x) -> int:
        count = 0
        for v in self.vocab_freq.values():
            if v < x:
                count += 1
        return count


    # Count num of all tokens
    def count_all_tokens(self) -> int:
        count = 0
        for v in self.vocab_freq.values():
            count += v
        return count
