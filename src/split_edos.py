import argparse
from typing import Any, List, Optional, Tuple
import random

from dataset_io import *


"""
Script to split input corpus into two parts.
Intended for creating train/dev or dev/test splits.
"""


def split(entries_in: List[Any]) -> Tuple[List[Any], List[Any], List[Any]]:
    train = []
    dev = []
    test = []
    for entry in entries_in:
        if entry['split'] == 'train':
            train.append(entry)
        elif entry['split'] == 'dev':
            dev.append(entry)
        elif entry['split'] == 'test':
            test.append(entry)
        else:
            raise Exception(f"Split not known: {entry['split']}")
    return train, dev, test


def main(args: argparse.Namespace) -> None:
    loader = LOADERS[args.loader]
    writer = WRITERS[args.writer]
    entries_in = loader.load(args.path_in)
    trainset, devset, testset = split(entries_in=entries_in)
    writer.write(trainset, args.path_train, header=list(entries_in[0].keys()))
    writer.write(devset, args.path_dev, header=list(entries_in[0].keys()))
    writer.write(testset, args.path_test, header=list(entries_in[0].keys()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path_in', help='Path to corpus that will be split.')
    parser.add_argument('--path_train', help='Output path for training set.')
    parser.add_argument('--path_dev', help='Output path for development set.')
    parser.add_argument('--path_test', type=int, help='Output path for test set.')
    parser.add_argument('-l', '--loader', help='The dataset loader to use (in dataset_io).')
    parser.add_argument('-w', '--writer', help='The dataset writer to use (in dataset_io).')
    parser.add_argument('-r', '--random_seed', type=int, 
                        help='Fix a random seed for reproducability.')
    cmd_args = parser.parse_args()
    random.seed(cmd_args.random_seed)
    main(cmd_args)
