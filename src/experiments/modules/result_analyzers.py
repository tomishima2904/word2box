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


# 刺激語の共通部分のboxと全ての語彙のboxとの類似度を計算
def dump_sim_scores(
        model,
        vocab_libs,
        words_list:List,
        dataloader,
        output_dir,
        device='cpu'
    ):

    ids_tensor: LongTensor = vocab_libs.words_list_to_ids_tensor(words_list).to(device)
    vocab_itos = vocab_libs.get_vocab_itos()

    for stimuli, stim_ids in zip(words_list, ids_tensor):
        result = []
        result.extend(stimuli)
        scores, labels = _compute_sim_with_vocab(stim_ids.to(device), dataloader, model)
        scores = scores.to('cpu').detach().numpy().tolist()
        labels = labels.to('cpu').detach().numpy().tolist()

        output_path = f"{output_dir}/{'_'.join(stimuli)}.csv"
        with open(output_path, 'w') as fo:
            stimuli_writer = csv.writer(fo)
            stimuli_writer.writerow(["labels", "scores"])
            output_list = [[vocab_itos[label], score] for label, score in zip(labels, scores)]
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
        LongTensor: Scores which are the most similar to intersection box of input `words`
        LongTensor: Labels which are the most similar to intersection box of input `words`
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


def summarize_sim_scores(
    output_dir,
    eval_file,
    words_list,
    vocab_libs,
    num_stimuli,
    num_output=300
):
    output_path = f"{output_dir}/{eval_file}.csv"
    with open(output_path, "w") as f:

        header = []
        for i in range(num_stimuli): header.append(f"stimulus_{i+1}")
        for i in range(num_stimuli): header.append(f"id_{i+1}")
        header.extend(["labels", "scores"])

        csvwriter = csv.writer(f)
        csvwriter.writerow(header)

        for stimuli in words_list:
            result = []
            result.extend(stimuli)
            stim_ids = vocab_libs.words_list_to_ids_tensor(words_list)
            result.extend(stim_ids.to('cpu').detach().numpy().tolist())

            sim_scores_path = f"{output_dir}/{'_'.join(stimuli)}.csv"
            labels_and_scores, _ = read_csv(sim_scores_path, has_header=True)
            labels_and_scores = labels_and_scores[:num_output]
            labels = [row[0] for row in labels_and_scores]
            scores = [row[1] for row in labels_and_scores]

            result.append(labels)
            result.append(scores)

            csvwriter.writerow(result)

    print(f"Successfully written {output_path} !")
