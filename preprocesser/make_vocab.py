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
