import json
import os
from os import path
import pickle
import multiprocessing as mp
from multiprocessing import Manager
import torch
import itertools
from collections import Counter

import torchtext
import torchtext.vocab as vocab
from torchtext.datasets import PennTreebank, WikiText2, WikiText103
from torch.utils.data import ConcatDataset, DataLoader

from ..datasets.word2vecgpu import LazyDatasetLoader, Word2VecDatasetOnDevice

from pathlib import Path
from typing import *


# global use_cuda
# use_cuda = torch.cuda.is_available()
# device = torch.cuda.current_device() if use_cuda else "cpu"
max_window = 10


def load_lines(dataset):
    eos_idxs = [i for i, token in enumerate(dataset) if token == "<eos>"]
    dataset_lines = [
        dataset[i + 1 : j + 1] for i, j in zip([-1] + eos_idxs, eos_idxs + [-2])
    ]
    return dataset_lines


def get_token_ids(dataset, vocab):
    dataset_tokenized = [
        [vocab.get(x, vocab["<unk>"]) for x in line if len(x.strip()) != 0]
        for line in dataset
    ]
    return dataset_tokenized


def load_vocab(data_dir: Union[str, Path]):
    vocab_tsv = Path(data_dir) / "vocab.tsv"
    if vocab_tsv.exists():
        vocab_stoi = {}
        vocab_freq = {}
        with vocab_tsv.open() as vocab_file:
            next(vocab_file)  # skips header line
            for token_id, line in enumerate(vocab_file):
                token, frequency = line.split()
                vocab_stoi[token] = int(token_id)
                vocab_freq[token] = int(frequency)
        return vocab_stoi, vocab_freq

    elif path.isfile(data_dir + "vocab_stoi.json") and path.isfile(
        data_dir + "vocab_freq.json"
    ):
        vocab_stoi = json.load(open(data_dir + "vocab_stoi.json", "r"))
        vocab_freq = json.load(open(data_dir + "vocab_freq.json", "r"))
        return vocab_stoi, vocab_freq

    else:
        train_txt = Path(data_dir) / "train.txt"
        vocab_freq = Counter()
        with train_txt.open() as train_txt_file:
            for tokens in _yield_tokens(train_txt_file):
                vocab_freq.update(tokens)

        specials = ['<unk>', '<pad>', '<eos>']
        vocab_obj = vocab.build_vocab_from_iterator([vocab_freq.keys()], specials=specials)
        vocab_stoi = vocab_obj.get_stoi()

        # Sort vocab_stoi by order of IDs
        vocab_stoi = {k: v for k, v in sorted(vocab_stoi.items(), key=lambda item: item[1])}

        vocab_stoi_file = open(data_dir + "/vocab_stoi.json", "w", encoding="utf-8")
        vocab_freq_file = open(data_dir + "/vocab_freq.json", "w", encoding="utf-8")
        json.dump(vocab_stoi, vocab_stoi_file, ensure_ascii=False)
        json.dump(vocab_freq, vocab_freq_file, ensure_ascii=False)
        vocab_stoi_file.close()
        vocab_freq_file.close()
        return vocab_stoi, vocab_freq


# See https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#codecell2
def _yield_tokens(data_iter):
    for line in data_iter:
        yield(line.split(' '))


def load_tokenizer(dataset):
    data_dir = "./data/" + dataset + "/"
    if path.isfile(data_dir + "train_tokenized.pkl"):
        train_tokenized = pickle.load(open(data_dir + "train_tokenized.pkl", "rb"))
    else:
        train_tokenized = []
        vocab_stoi = json.load(open(data_dir + "vocab_stoi.json", "r"))
        vocab_freq = json.load(open(data_dir + "vocab_freq.json", "r"))

        with open(data_dir + "train.txt", "r") as f:
            for line in f:
                words = line.split()
                train_tokenized.append(
                    [vocab_stoi[ele] for ele in words] + [vocab_stoi["<eos>"]]
                )

        pickle.dump(train_tokenized, open(data_dir + "train_tokenized.pkl", "wb"))
    return train_tokenized


def load_train_data_as_tensor(dataset):
    data_dir = "./data/" + dataset + "/"
    tensor_file = Path(data_dir + "train.pt")
    ex_tensor_file = Path(data_dir + "example.pt")
    if tensor_file.exists():
        return torch.load(tensor_file)
    else:
        train_tensor = torch.tensor(
            list(itertools.chain.from_iterable(load_tokenizer(dataset)))
        )
        torch.save(train_tensor, tensor_file)
        torch.save(train_tensor[:500000], ex_tensor_file)
        print("train.pt has been dumped successfully")
    return train_tensor


# Original func but I don't use
# Because I splited this func into `get_vocab` and `get_train_iter`
def get_iter_on_device(
    batch_size,
    dataset,
    model_type,
    n_gram,
    subsample_thresh,
    data_device,
    add_pad,
    eos_mask,
    ignore_unk,
):
    print("Loading VOCAB & Tokenized Training files ...")
    vocab_stoi, vocab_freq = load_vocab("./data/" + dataset)
    train_tokenized = load_train_data_as_tensor(dataset)

    ## Create Vocabulary properties
    print("Creating iterable dataset ...")
    vocab_itos = [k for k, v in sorted(vocab_stoi.items(), key=lambda item: item[1])]

    vocab = {
        "stoi": vocab_stoi,
        "freqs": vocab_freq,
        "itos": vocab_itos,
    }

    # Since we won't train on <pad> and <eos>. These should not come in any sort of
    # subsampling and negative sampling part.
    vocab_freq["<pad>"] = 0
    vocab_freq["<unk>"] = 0

    if eos_mask:
        vocab_freq["<eos>"] = 0
    # We want to pad max window length pad tokens and eos to the start
    # and to the end of the corpus and remove <unk> tokens

    # if add_pad:
    #     paddings = torch.tensor([TEXT.stoi['<eos>']] * max_window)
    #     train_tokenized = torch.cat(
    #                         (paddings,
    #                         train_tokenized[train_tokenized != TEXT.stoi['<unk>']],
    #                         paddings))

    ## Create data on the device
    print("Creating iterable dataset on GPU/CPU...")

    train_iter = LazyDatasetLoader(
        training_tensor=train_tokenized,
        n_splits=1000,
        window_size=n_gram,
        vocab=vocab,
        subsample_thresh=subsample_thresh,
        eos_mask=eos_mask,
        device=data_device,
        batch_size=batch_size,
        ignore_unk=ignore_unk,
    )

    val_iter, test_iter = None, None
    return vocab, train_iter, val_iter, test_iter, None


def get_vocab(dataset, eos_mask):
    print("Loading VOCAB & Tokenized Training files ...")
    vocab_stoi, vocab_freq = load_vocab("./data/" + dataset)

    ## Create Vocabulary properties
    print("Creating iterable dataset ...")
    vocab_itos = [k for k, v in sorted(vocab_stoi.items(), key=lambda item: item[1])]

    vocab = {
        "stoi": vocab_stoi,
        "freqs": vocab_freq,
        "itos": vocab_itos,
    }

    # Since we won't train on <pad> and <eos>. These should not come in any sort of
    # subsampling and negative sampling part.
    vocab_freq["<pad>"] = 0
    vocab_freq["<unk>"] = 0

    if eos_mask:
        vocab_freq["<eos>"] = 0

    return vocab


def get_train_iter(
        batch_size,
        dataset,
        model_type,
        n_gram,
        subsample_thresh,
        data_device,
        add_pad,
        eos_mask,
        ignore_unk,
        vocab,
    ):
    train_tokenized = load_train_data_as_tensor(dataset)
    train_iter = LazyDatasetLoader(
        training_tensor=train_tokenized,
        n_splits=1000,
        window_size=n_gram,
        vocab=vocab,
        subsample_thresh=subsample_thresh,
        eos_mask=eos_mask,
        device=data_device,
        batch_size=batch_size,
        ignore_unk=ignore_unk,
    )
    return train_iter
