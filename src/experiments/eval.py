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

from language_modeling_with_boxes.models import \
    Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss

from experiments.utils import file_handlers as fh
from datasets import TrainedAllVocabDataset
from experiments.modules.vocab_libs import VocabLibs, VocabLibsWithFreq
from experiments.modules import result_analyzers as ra


def eval(args):
    # 保存してあるモデルの設定をロードする
    config = json.load(open(args.result_dir + "/config.json", "r"))

    # itos (IDから文字列) の辞書を作成
    vocab_libs = VocabLibsWithFreq(f"data/{config['dataset']}/vocab_stoi.json",
                                   f"data/{config['dataset']}/vocab_freq.json")
    vocab_size = vocab_libs.get_vocab_size()

    # モデルのインスタンスを作成する
    model = Word2BoxConjunction(vocab_size=vocab_size,
                                embedding_dim=config["embedding_dim"],
                                batch_size=config["batch_size"],
                                n_gram=config["n_gram"],
                                intersection_temp=config["int_temp"],
                                volume_temp=config["vol_temp"],
                                box_type=config["box_type"])

    # 作成したインスタンスに訓練済みモデルのパラメータを読み込む
    print("Loading model ...")
    model_ckpt = torch.load(args.result_dir + "/best_model.ckpt", map_location='cpu')
    model.load_state_dict(model_ckpt['model_state_dict'])

    # GPUを使用して計算
    if "cuda" in args.data_device:
        device = torch.device(args.data_device if torch.cuda.is_available() else "cpu")
        print(f"Using {device}")
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"

    if args.multi_gpu==1 and "cuda" in args.data_device:
        pass
        # model = DDP(model, device_ids=[0, 1])
        # model = model.module

    model.to(device)

    # 語彙のデータローダーを作成
    dataloader = DataLoader(dataset= TrainedAllVocabDataset(vocab_libs.vocab_stoi, model, device),
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory =bool(args.pin_memory))

    # 出力用のディレクトリがなければ作成
    output_dir = f"{args.result_dir}/analysis"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # 訓練済み埋め込み表現のboxのvolumeを計算
    ra.compute_allbox_vols(vocab_libs, dataloader, model.box_type,
                           device, output_dir, dist_type=args.dist_type,
                           temp=config["vol_temp"], gumbel_beta=config["int_temp"])


    # 以下では評価用データセットを利用
    # 評価用データセットをロード
    dataset_dir = 'data/qualitative_datasets'
    words_list = fh.read_csv(dataset_dir + "/" + args.eval_file + ".csv", has_header=False)
    words = words_list[0]

    # 出力用のディレクトリがなければ作成
    output_dir = f"{args.result_dir}/{args.eval_file}"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # 訓練済みのBoxを出力
    ra.dump_boxes_zZ(model, vocab_libs, words, output_dir)
    if model.box_type in ("CenterBoxTensor", "CenterSigmoidBoxTensor"):
        ra.dump_boxes_cenoff(model, vocab_libs, words, output_dir)
        if args.w2v_dir is not None:
            ra.dump_centers_with_w2v(model, vocab_libs, words, args.w2v_dir, output_dir)

    # 全語彙との類似度を計算してdump
    ra.dump_sim_scores(model, vocab_libs, words_list, dataloader, output_dir, device=device)

    # 刺激語の数
    if type(words_list[0]) == list:
        num_stimuli = len(words_list[0])
    else:
        num_stimuli = len(words_list)

    # dumpされたデータを見やすくするために要約
    ra.summarize_sim_scores(vocab_libs, words_list, output_dir, args.eval_file, num_stimuli, args.num_output)

    # boxを次元毎にプロット
    for words in words_list:
        print(words)
        ra.plot_eachdim_of_boxes(model, vocab_libs, words, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16384, help="batch size for evaluating with all vocab")
    parser.add_argument('--data_device', type=str, default="cpu", help="device type")
    parser.add_argument('--dist_type', type=str, default="logsoft", help="[logsoft, relu, abs]")
    parser.add_argument('--eval_file', type=str, required=True, help="file path where the eval file is")
    parser.add_argument('--multi_gpu', type=int, default=0, help="1 if you use multi gpu")
    parser.add_argument('--num_output', type=int, default=300, help="number of labels and scores to be output")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers for dataloader")
    parser.add_argument('--output_allscores', type=int, default=1)
    parser.add_argument('--pin_memory', type=int, default=0, help="1 if you use pin memory in dataloader")
    parser.add_argument('--result_dir', type=str, required=True, help="dir path where the saved model is")
    parser.add_argument('--w2v_dir', type=str, default=None, help="dir path where the trained word2vec model is")
    args = parser.parse_args()

    eval(args)
