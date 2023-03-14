import torch
from torch import Tensor, LongTensor
from typing import Union, List, Dict
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

from ..utils.file_handlers import *
from vocab_libs import VocabLibs, VocabLibsWithFreq


# Plot similarity
def plot_similarity(input_dir, stimulus:str, vocab_freq:Dict, title="x"):

    # Read similarity from csv file
    similairties_list, header = read_csv(f"{input_dir}/{stimulus}.csv", has_header=True)
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

    savefile = f"{input_dir}/{stimulus}.png"
    fig.savefig(savefile)
    print("Plotting has done")


def compute_allbox_volumes(model, vocab_libs: VocabLibsWithFreq, output_dir, dist_type: str="abs"):
    vocab_size = len(vocab_libs.get_vocab_size())
    volume_list = torch.zeros(vocab_size, dtype=torch.long)
    vocab_itos = vocab_libs.get_vocab_itos()

    with torch.no_grad():
        for i in tqdm(torch.arange(vocab_size), total=len(vocab_size)):
            emb = model.embeddings_word(i)
            emb_diff = emb.Z-emb.z
            if dist_type == "relu":
                emb_diff = torch.relu(emb_diff)
                filename = "largest_relu"
            elif dist_type == "abs":
                emb_diff = torch.abs(emb_diff)
                filename = "largest_abs"
            volume_list[i] = torch.sum(emb_diff)
        sorted_emb, indicies = torch.sort(LongTensor(volume_list), descending=True)
        sorted_emb = sorted_emb.to('cpu').detach().numpy().copy()
        indicies = indicies.to('cpu').detach().numpy().copy()

    if not os.path.isdir(f"{output_dir}/analysis"):
        os.makedirs(f"{output_dir}/analysis")

    with open(f"{output_dir}/analysis/{filename}.csv", "w") as f:
        csv_writer_ = csv.writer(f)
        csv_writer_.writerow(["id", "word", "volume", "freq"])
        for emb, i in tqdm(zip(sorted_emb, indicies)):
            if vocab_itos[i] == "<pad>": continue
            if vocab_itos[i] == "<eos>": continue
            csv_writer_.writerow([i, vocab_itos[i], emb, vocab_libs.vocab_freq[vocab_itos[i]]])
