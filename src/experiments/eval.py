import torch
import torchtext
from torch import Tensor, LongTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import pickle, json
import sys, os
from tqdm import tqdm
from typing import Union, List, Dict

import argparse
from pathlib import Path


# モデルのリロードに必要なモジュールをインポートする
from language_modeling_with_boxes.models import Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss

import experiments.set_operation as set_operation
from experiments.utils.file_handlers import *
from datasets import TrainedAllVocabDataset
from experiments.modules.vocab_libs import VocabLibs


def eval(args):

    if "cuda" in args.data_device:
        device = torch.device(args.data_device if torch.cuda.is_available() else "cpu")
        print(f"Using {device}")
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"

    # 保存してあるモデルの設定をロードする
    config = json.load(open(args.result_dir + "/config.json", "r"))

    # itos (IDから文字列) の辞書を作成
    print("Loading vocab file...")
    vocab_stoi = json.load(open("data/" + config["dataset"] + "/vocab_stoi.json", "r"))
    vocab_libs = VocabLibs(vocab_stoi)
    vocab_itos = vocab_libs.get_vocab_itos()

    # モデルのインスタンスを作成する
    model = Word2BoxConjunction(
        vocab_size=len(vocab_stoi),
        embedding_dim=config["embedding_dim"],
        batch_size=config["batch_size"],
        n_gram=config["n_gram"],
        intersection_temp=config["int_temp"],
        volume_temp=config["vol_temp"],
        box_type=config["box_type"],
    )

    # 作成したインスタンスに訓練済みモデルのパラメータを読み込む
    print("Loading model...")
    model_ckpt = torch.load(args.result_dir + "/best_model.ckpt", map_location='cpu')
    model.load_state_dict(model_ckpt['model_state_dict'])

    if args.multi_gpu==1 and "cuda" in args.data_device:
        pass
        # model = DDP(model, device_ids=[0, 1])
        # model = model.module

    # 語彙のデータローダー
    dataloader = DataLoader(
        dataset= TrainedAllVocabDataset(vocab_stoi, model, device),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory =bool(args.pin_memory),
    )
    model.to(device)

    # 評価用データセットをロード
    dataset_dir = 'data/qualitative_datasets'
    eval_words_list = read_csv(dataset_dir + "/" + args.eval_file + ".csv", has_header=False)
    eval_ids_list: LongTensor = vocab_libs.words_list_to_ids_tensor(eval_words_list).to(device)
    assert len(eval_words_list) == len(eval_ids_list), f"cat't match the length of `words_list` {len(eval_words_list)} and `ids_list` {len(eval_ids_list)}"

    output_dir = f"{args.result_dir}/{args.eval_file}"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # 刺激語の共通部分のboxと全ての語彙のboxとの類似度を計算
    with open(f"{output_dir}/{args.eval_file}.csv", "w") as f:

        num_stimuli = eval_ids_list.size(-1)
        header = []
        for i in range(num_stimuli): header.append(f"stimulus_{i+1}")
        for i in range(num_stimuli): header.append(f"id_{i+1}")
        header.extend(["labels", "scores"])

        csv_writer = csv.writer(f)
        csv_writer.writerow(header)

        for stimuli, stim_ids in zip(eval_words_list, eval_ids_list):
            result = []
            result.extend(stimuli)
            scores, labels = set_operation.all_words_similarity(stim_ids.to(device), dataloader, model)
            scores = scores.to('cpu').detach().numpy().tolist()
            labels = labels.to('cpu').detach().numpy().tolist()

            if args.output_allscores:
                with open(f"{output_dir}/{stimuli[0]}.csv", 'w') as fo:
                    stimuli_writer = csv.writer(fo)
                    stimuli_writer.writerow(["labels", "scores"])
                    output_list = [[vocab_itos[label], score] for label, score in zip(labels, scores)]
                    stimuli_writer.writerows(output_list)

            scores = scores[:args.num_output]
            labels = labels[:args.num_output]

            result.extend(stim_ids.to('cpu').detach().numpy().tolist())
            similar_words = [vocab_itos[label] for label in labels]
            result.append(similar_words)
            result.append(scores)
            csv_writer.writerow(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True, help="dir path where saved model is")
    parser.add_argument('--data_device', type=str, default="cpu", help="device type")
    parser.add_argument('--eval_file', type=str, required=True, help="file path where eval file is")
    parser.add_argument('--batch_size', type=int, default=16384, help="batch size for evaluating on all vocab")
    parser.add_argument('--num_output', type=int, default=300, help="number of labels and scores to be output")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers for dataloader")
    parser.add_argument('--output_allscores', type=int, default=1)
    parser.add_argument('--pin_memory', type=int, default=0, help="1 if you use pin memory in dataloader")
    parser.add_argument('--multi_gpu', type=int, default=0, help="1 if you use multi gpu")
    args = parser.parse_args()

    eval(args)
