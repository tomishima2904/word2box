import torch
from torch import Tensor, LongTensor
from torch.utils.data import DataLoader
from typing import Union, List, Dict
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

from language_modeling_with_boxes.models import Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss
from language_modeling_with_boxes.box.box_wrapper import BoxTensor

from ..utils.file_handlers import *
from .vocab_libs import VocabLibs, VocabLibsWithFreq
from set_operations import SetOperations


# Plot similarity
def plot_similarity(save_dir, stimulus:str, vocab_freq:Dict, title="x"):

    # Read similarity from csv file
    similairties_list, header = read_csv(f"{save_dir}/{stimulus}.csv", has_header=True)
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

    savefile = f"{save_dir}/{stimulus}.png"
    fig.savefig(savefile)
    print("Plotting has done")


def compute_allbox_volumes(model, vocab_libs: VocabLibsWithFreq, output_dir, dist_type: str="abs"):
    vocab_size:int = vocab_libs.get_vocab_size()
    volume_list = torch.zeros(vocab_size, dtype=torch.long)
    vocab_itos:List = vocab_libs.get_vocab_itos()

    with torch.no_grad():
        for i in tqdm(torch.arange(vocab_size), total=vocab_size):
            emb = model.embeddings_word(i)
            emb_diff = emb.Z-emb.z
            if dist_type == "relu":
                emb_diff = torch.relu(emb_diff)
            elif dist_type == "abs":
                emb_diff = torch.abs(emb_diff)
            volume_list[i] = torch.sum(emb_diff)
        sorted_emb, indicies = torch.sort(LongTensor(volume_list), descending=True)
        sorted_emb = sorted_emb.to('cpu').detach().numpy().copy()
        indicies = indicies.to('cpu').detach().numpy().copy()

    if dist_type == "relu": filename = "largest_relu"
    elif dist_type == "abs": filename = "largest_abs"
    output_path = f"{output_dir}/{filename}.csv"

    with open(output_path, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["id", "word", "volume", "freq"])
        for emb, i in tqdm(zip(sorted_emb, indicies)):
            if vocab_itos[i] == "<pad>": continue
            if vocab_itos[i] == "<eos>": continue
            csvwriter.writerow([i, vocab_itos[i], emb, vocab_libs.vocab_freq[vocab_itos[i]]])
        print(f"Successfully written {output_path} !")

    plot_allbox_volumes(output_dir, filename)


# Plot volume of boxes by descending order
def plot_allbox_volumes(save_dir, filename):
    with open(f"{save_dir}/{filename}.csv", "r") as f:
        csvreader = csv.reader(f)
        header = next(csvreader)
        volumes = [float(row[2]) for row in csvreader]
        x = range(len(volumes))

        fig, ax = plt.subplots()
        ax.plot(x, volumes)
        ax.set_title("Largest boxes")
        fig.savefig(f"{save_dir}/{filename}.png")
        print("Plotting has done")

