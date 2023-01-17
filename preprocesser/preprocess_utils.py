import re
import json
import gzip
import argparse
from collections import OrderedDict

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


def drop_words(args, max_freq=2):
    print(f"Dropping words occurring less than {max_freq} times")
    with open(args.output_dir + "/vocab_freq.json", "r") as f:
        vocab_freq = json.load(f)
        drop_words_list = [key for key, value in vocab_freq.items() if value < max_freq]

    with open(args.output_dir + "/train.txt", "r") as fi, \
         open(args.output_dir + "/dropped_train.txt", "w") as fo:
        for line in fi:
            text = line.split()
            text = [word for word in text if word not in drop_words_list]
            print(" ".join(text), file=fo)
        print("Successfully dumped dropped train file !")
