import torch
from torch import Tensor, LongTensor
from torch.utils.data import DataLoader
from typing import Union, List, Dict, Tuple
import json
import csv
import matplotlib.pyplot as plt
import japanize_matplotlib
from tqdm import tqdm
import numpy as np
import os

from language_modeling_with_boxes.models import \
    Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss
from language_modeling_with_boxes.box.box_wrapper import BoxTensor

from ..utils import file_handlers as fh
from .vocab_libs import VocabLibs, VocabLibsWithFreq
from set_operations import SetOperations


# Plot similarity
def plot_similarity(save_dir, stimulus:str, vocab_freq:Dict, title="x"):

    # Read similarity from csv file
    similairties_list, header = fh.read_csv(f"{save_dir}/{stimulus}.csv",
                                            has_header=True)
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

    ax1.set_title(f"Similarity of {stimulus}")
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


def compute_allbox_volumes(
        vocab_libs: VocabLibsWithFreq,
        dataloader,
        box_type,
        device,
        output_dir,
        dist_type: str="abs"
    ):

    volume_list = []
    vocab_itos:List = vocab_libs.get_vocab_itos()

    with torch.no_grad():

        # Compute volumes of all boxes
        for boxes, labels in tqdm(dataloader):
            B = len(boxes)

            if box_type in ("CenterBoxTensor", "CenterSigmoidBoxTensor"):
                offset = boxes[..., 1, :].to(device)
                emb_diffs = offset * 2
            else:
                z = boxes[..., 0, :].to(device)
                Z = boxes[..., 1, :].to(device)
                emb_diffs = Z - z

            if dist_type == "relu":
                emb_diffs = torch.relu(emb_diffs)
            elif dist_type == "abs":
                emb_diffs = torch.abs(emb_diffs)
            else:
                raise ValueError(f"Invalid distance type {dist_type}")

            volume_list.extend(torch.sum(emb_diffs, dim=-1).to('cpu'))

        # Sort volumes by descending order
        volume_list = Tensor(volume_list, device='cpu')
        sorted_emb, indicies = torch.sort(volume_list, descending=True)
        sorted_emb = sorted_emb.to('cpu').detach().numpy().copy()
        indicies = indicies.to('cpu').detach().numpy().copy()

    # Set output file name
    if dist_type == "relu":
        filename = "largest_relu"
    elif dist_type == "abs":
        filename = "largest_abs"
    output_path = f"{output_dir}/{filename}.csv"

    # Dump volumes
    with open(output_path, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["id", "word", "volume", "freq"])
        for emb, i in tqdm(zip(sorted_emb, indicies)):
            if vocab_itos[i] == "<pad>": continue
            if vocab_itos[i] == "<eos>": continue
            row = [i, vocab_itos[i], emb, vocab_libs.vocab_freq[vocab_itos[i]]]
            csvwriter.writerow(row)
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


# 刺激語の共通部分のboxと全ての語彙のboxとの類似度を計算
def dump_sim_scores(
        model,
        vocab_libs,
        words_list:List,
        dataloader,
        output_dir,
        device='cpu',
    ):

    ids_tensor: LongTensor = vocab_libs.words_list_to_ids_tensor(words_list).to(device)
    vocab_itos = vocab_libs.get_vocab_itos()

    for stimuli, stim_ids in zip(words_list, ids_tensor):
        result = []
        result.extend(stimuli)
        scores, labels = _compute_sim_with_vocab(stim_ids.to(device), dataloader, model)
        scores = scores.to('cpu').detach().numpy().tolist()
        labels = labels.to('cpu').detach().numpy().tolist()

        output_path = f"{output_dir}/sims_{'_'.join(stimuli)}.csv"
        with open(output_path, 'w') as fo:
            stimuli_writer = csv.writer(fo)
            stimuli_writer.writerow(["labels", "scores"])
            output_list = [[vocab_itos[label], score] \
                           for label, score in zip(labels, scores)]
            stimuli_writer.writerows(output_list)
            print(f"Successfully written {output_path} !")


def _compute_sim_with_vocab(
        word_ids: LongTensor,
        dataloader: DataLoader,
        model: Union[Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss],
    ) -> Tuple[LongTensor, LongTensor]:
    """ Output similar scores and labels to box of input words

    Args:
        words (LongTensor): 刺激語IDのリスト. List of indices of words.
        dataloader (DataLoader): Vocabulary.
        model : Trained model.

    Returns:
        Tensor: Similarity scores by descending order
        LongTensor: Labels of similarity scores
    """

    device = word_ids.device

    with torch.no_grad():

        intersection_box = SetOperations.intersect_multiple_box(word_ids, model)  # [1, 1, 2, embedding_dim]
        all_scores = Tensor([])
        all_labels = LongTensor([])
        box_type = model.box_type

        for boxes, labels in tqdm(dataloader):

            # 共通部分と語彙のBoxTensorを作成
            # Make BoxTensors
            B = len(boxes)
            if box_type in ("CenterBoxTensor", "CenterSigmoidBoxTensor"):
                center = boxes[..., 0, :].unsqueeze(-2).to(device)
                offset = boxes[..., 1, :].unsqueeze(-2).to(device)
                vocab_z: Tensor = center - offset
                vocab_Z: Tensor = center + offset
            else:
                vocab_z: Tensor = boxes[..., 0, :].unsqueeze(-2).to(device)
                vocab_Z: Tensor = boxes[..., 1, :].unsqueeze(-2).to(device)
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
            assert scores.size() == labels.size(), \
                f"can't match size of `scores {scores.size()}` and `labels {labels.size()}`"
            all_scores = torch.cat([all_scores, scores])
            all_labels = torch.cat([all_labels, labels.to('cpu').detach()])

        # scores を降順に並び替え、それに伴い labels も並び替える
        # Sort scores in descending order, also sort labels along with scores
        all_scores, sorted_indices = torch.sort(all_scores, descending=True)
        all_labels = all_labels[sorted_indices]

    return all_scores, all_labels


def summarize_sim_scores(
        vocab_libs,
        words_list,
        output_dir,
        eval_file,
        num_stimuli,
        num_output=300,
        with_id=False,
    ):
    output_path = f"{output_dir}/{eval_file}.csv"
    with open(output_path, "w") as f:

        header = []
        for i in range(num_stimuli): header.append(f"stimulus_{i+1}")
        if with_id:
            for i in range(num_stimuli): header.append(f"id_{i+1}")
        header.extend(["labels", "scores"])

        csvwriter = csv.writer(f)
        csvwriter.writerow(header)

        for stimuli in words_list:
            result = []
            if with_id:
                stim_ids = vocab_libs.words_list_to_ids_tensor(stimuli)
                result.extend(stim_ids.to('cpu').detach().numpy().tolist())
            result.extend(stimuli)

            sim_scores_path = f"{output_dir}/{'_'.join(stimuli)}.csv"
            labels_and_scores, _ = fh.read_csv(sim_scores_path, has_header=True)
            labels_and_scores = labels_and_scores[:num_output]
            labels = [row[0] for row in labels_and_scores]
            scores = [row[1] for row in labels_and_scores]

            result.append(labels)
            result.append(scores)

            csvwriter.writerow(result)

    print(f"Successfully written {output_path} !")


def dump_boxes_zZ(
        model,
        vocab_libs,
        words: List,  # This arg should be 1-dim list
        output_dir,
        output_file="boxes_zZ.csv",
    ):
    model.to('cpu')

    # Embed words_list
    ids_tensor: LongTensor = vocab_libs.words_list_to_ids_tensor(words).to('cpu')
    word_embs = model.embeddings_word(ids_tensor)
    all_z = word_embs.z
    all_z = torch.t(all_z).to('cpu').detach().numpy()
    all_Z = word_embs.Z
    all_Z = torch.t(all_Z).to('cpu').detach().numpy()

    results = np.zeros([all_z.shape[0], 2 * all_z.shape[-1]])
    results[..., 0::2] = all_z
    results[..., 1::2] = all_Z
    results = results.tolist()

    # Make header
    labels = []
    for word in words:
        labels.append(f"{word}-")
        labels.append(f"{word}+")

    assert len(results[0]) == len(labels), \
        f"len(results[0])=={len(results[0])}, len(labels)=={len(labels)}"

    # Make a dir if not exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Write boxes
    output_path = f"{output_dir}/{output_file}"
    fh.write_csv(output_path, results, header=labels)
    print(f"Successfully written {output_path} !")


def dump_boxes_cenoff(
        model,
        vocab_libs,
        words: List,  # This arg should be 1-dim list
        output_dir,
        output_file="boxes_cenoff.csv",
    ):
    assert model.box_type in ("CenterBoxTensor", "CenterSigmoidBoxTensor"), \
        "Box type should be `CenterBoxTensor` or `CenterSigmoidBoxTensor`"

    model.to('cpu')

    # Embed words_list
    ids_tensor: LongTensor = vocab_libs.words_list_to_ids_tensor(words).to('cpu')
    word_embs = model.embeddings_word(ids_tensor)
    all_cen = word_embs.center
    all_cen = torch.t(all_cen).to('cpu').detach().numpy()
    all_off = word_embs.offset
    all_off = torch.t(all_off).to('cpu').detach().numpy()

    results = np.zeros([all_cen.shape[0], 2 * all_cen.shape[-1]])
    results[..., 0::2] = all_cen
    results[..., 1::2] = all_off
    results = results.tolist()

    # Make header
    labels = []
    for word in words:
        labels.append(f"{word}Ct")
        labels.append(f"{word}Of")

    assert len(results[0]) == len(labels), \
        f"len(results[0])=={len(results[0])}, len(labels)=={len(labels)}"

    # Make a dir if not exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Write boxes
    output_path = f"{output_dir}/{output_file}"
    fh.write_csv(output_path, results, header=labels)


def plot_eachdim_of_boxes(
        model,
        vocab_libs,
        words: List,  # This arg should be 1-dim list
        output_dir,
        output_file: str=None,
    ):

    model.to('cpu')

    # Embed words
    ids_tensor: LongTensor = vocab_libs.words_list_to_ids_tensor(words).to('cpu')
    word_embs = model.embeddings_word(ids_tensor)
    zs = word_embs.z.to('cpu').detach().numpy()
    Zs = word_embs.Z.to('cpu').detach().numpy()

    # Set properties
    fig, ax = plt.subplots()
    dim = zs.shape[-1]
    colors = ["blue", "red", "green", "yellow", "pink"]
    assert len(words) < len(colors), "Number of colors is not enough"
    for i, (z, Z) in enumerate(zip(zs, Zs)):
        isplotted: bool = False
        for d in range(dim):
            if Z[d] > z[d]:
                if isplotted:
                    ax.bar(d, Z[d]-z[d], bottom=z[d], color=colors[i],
                           alpha=0.5, align='center')
                else:
                    ax.bar(d, Z[d]-z[d], bottom=z[d], color=colors[i],
                           alpha=0.5, align='center', label=words[i])
                    isplotted = True

    ax.set_xlabel('Dimensions')
    ax.set_ylabel('Box position')
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(words), loc='upper center')

    # Output file
    if output_file is None:
        output_file = f"boxes_{('_').join(words)}.png"
    output_path = f"{output_dir}/{output_file}"
    fig.savefig(output_path)
    print(f"Successfully plotted {output_path}")


def dump_centers_with_w2v(
        model,
        vocab_libs,
        words: List,  # This arg should be 1-dim list
        w2v_dir,
        output_dir,
        w2v_file="all_vectors.txt",
        output_file="centers_with_w2v.csv",
    ):
    assert model.box_type in ("CenterBoxTensor", "CenterSigmoidBoxTensor"), \
        "Box type should be `CenterBoxTensor` or `CenterSigmoidBoxTensor`"

    # Embed words with BoxEmbedding
    ids_tensor: LongTensor = vocab_libs.words_list_to_ids_tensor(words).to('cpu')
    word_embs = model.embeddings_word(ids_tensor)
    all_cen = word_embs.center
    all_cen = torch.t(all_cen).to('cpu').detach().numpy()

    # Embed words with Word2Vec
    word2vec_model = load_word2vec(w2v_dir, w2v_file, model_type="torch")
    w2v_embs = word2vec_model[ids_tensor]
    w2v_embs = torch.t(w2v_embs).to('cpu').detach().numpy()

    assert all_cen.shape == w2v_embs.shape, \
        f"all_cen.size()=={all_cen.shape}, w2v_embs.size()=={w2v_embs.shape}"

    # 奇数列に中心ベクトル, 偶数列にWord2Vecのベクトルを割り当てる
    embeddings = np.zeros([all_cen.shape[0], 2*all_cen.shape[1]])
    embeddings[:, 0::2] = w2v_embs
    embeddings[:, 1::2] = all_cen

    # Make header
    labels = []
    for word in words:
        labels.append(f"{word}W2V")
        labels.append(f"{word}Ct")

    # Make a dir if not exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Write embeddings
    output_path = f"{output_dir}/{output_file}"
    fh.write_csv(output_path, data=embeddings, header=labels)


def load_word2vec(w2v_dir, w2v_file="all_vectors.txt", model_type="list"):
    print("Loading trained Word2Vec model ...")
    w2v_path = f"{w2v_dir}/{w2v_file}"
    with open(w2v_path, 'r') as f:
        header = next(f).split(' ')
        embeddings = f.readlines()
        vocab_size = header[0]

        embeddings = [embedding.split(' ')[1:] for embedding in embeddings]
        embeddings = np.array(embeddings, dtype=np.float32)
        if model_type == "numpy":
            return embeddings

        if model_type == "list":
            return embeddings.tolist()

        embeddings = torch.from_numpy(embeddings)
        if model_type == "torch":
            return Tensor(embeddings)
        else:
            raise ValueError(f"Invalid model type {model_type}")
