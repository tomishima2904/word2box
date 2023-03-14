import torch
from torch import Tensor, LongTensor
from typing import Union, List, Dict
import json
import matplotlib.pyplot as plt
from matplotlib import cm

from ..utils.file_handlers import *


class ResultReader(object):
    def __init__(self, result_dir: str) -> None:
        self.result_dir = result_dir


    # Plot similarity
    def plot_similarity(self, stimulus:str, vocab_freq:Dict, title="x"):

        # Read similarity from csv file
        similairties_list, header = read_csv(f"{self.result_dir}/{stimulus}.csv", has_header=True)
        words = [row[0] for row in similairties_list]
        scores = [float(row[1]) for row in similairties_list]

        # Drop <pad> and <eos>
        pad_index = words.index('<pad>')
        words.pop(pad_index)
        scores.pop(pad_index)
        eos_index = words.index('<eos>')
        words.pop(eos_index)
        scores.pop(eos_index)

        freqs = [vocab_freq[word] for word in words]
        word_index = words.index(stimulus)
        x = range(len(words))

        print("Plotting...")
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(x, scores, label="Similarity", color='orange')
        ax2.plot(x, freqs, label="Frequency", color='blue')
        #ax1.axvline(x=word_index, color='red')
        ax1.scatter(x=word_index, y=scores[word_index], color='red')

        ax1.set_title(f"Similarity of {title}")
        ax1.set_xlabel("Words")
        ax1.set_ylabel('Similarity')
        ax2.set_ylabel('Frequency')

        handler1, label1 = ax1.get_legend_handles_labels()
        handler2, label2 = ax2.get_legend_handles_labels()
        ax1.legend(handler1 + handler2, label1 + label2, borderaxespad=0.)
        # ax1.set_yscale('log')

        savefile = f"{self.result_dir}/{stimulus}.png"
        fig.savefig(savefile)
        print("Plotting has done")
