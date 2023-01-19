import re
import json
import gzip
import argparse
from collections import OrderedDict
import os

import torchtext


def make_vocab(args):
    print("Maiking vocab files ......")
    TEXT = torchtext.data.Field()
    train_split = torchtext.datasets.LanguageModelingDataset.splits(
        path=args.output_dir,
        train="train.txt",
        validation=None,
        test=None,
        text_field=TEXT,
    )
    TEXT.build_vocab(train_split[0])
    vocab_stoi_file = open(args.output_dir + "/vocab_stoi.json", "w", encoding="utf-8")
    vocab_freq_file = open(args.output_dir + "/vocab_freq.json", "w", encoding="utf-8")
    json.dump(TEXT.vocab.stoi, vocab_stoi_file, ensure_ascii=False)
    json.dump(TEXT.vocab.freqs, vocab_freq_file, ensure_ascii=False)
    vocab_stoi_file.close()
    vocab_freq_file.close()
    print("Successfully dumped vocab files !")


def replace_oov(args, min_count=5, renew=False):
    print(f"Replacing words occurring less than {min_count} times to <unk>")
    with open(args.output_dir + "/vocab_freq.json", "r") as f:
        vocab_freq = json.load(f)
        # drop_words_list = [key for key, value in vocab_freq.items() if value < min_count]s

    with open(args.output_dir + "/train.txt", "r") as fi, \
         open(args.output_dir + "/replaced_train.txt", "w") as fo:
        for line in fi:
            text = line.split()
            text = [word if vocab_freq[word] >= min_count else "<unk>" for word in text]  # OOV is replaced to <unk>
            print(" ".join(text), file=fo)
        print("Successfully dumped train file !")

    # If renew is true, original train file is replaced to preprocessed train file
    if renew:
        os.remove(args.output_dir + "/train.txt")
        os.rename(args.output_dir + "/replaced_train.txt", args.output_dir + "/train.txt")
        print("Renewed train file !")
