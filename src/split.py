import argparse
from typing import Any, List, Optional, Tuple
import random

from dataset_io import *


"""
Script to split in input corpus into two parts.
Intended for creating train/dev or dev/test splits.
"""


def split(entries_in: List[Any], num_a: Optional[int], ratio: Optional[float]
          ) -> Tuple[List[Any], List[Any]]:
    random.shuffle(entries_in)
    if not num_a:
        assert ratio is not None
        num_entries = len(entries_in)
        num_a = int(round(num_entries * ratio, 0))
    entries_a = entries_in[:num_a]
    entries_b = entries_in[num_a:]
    return entries_a, entries_b


def main(args: argparse.Namespace) -> None:
    loader = LOADERS[args.loader]
    writer = WRITERS[args.writer]
    entries_in = loader.load(args.path_in)
    entries_a, entries_b = split(entries_in=entries_in, num_a=args.num_a, ratio=args.ratio)
    writer.write(entries_a, args.path_out_a, header=list(entries_in[0].keys()))
    writer.write(entries_b, args.path_out_b, header=list(entries_in[0].keys()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path_in', help='Path to corpus that will be split.')
    parser.add_argument('-a', '--path_out_a', help='Output path for part-a of the produced split.')
    parser.add_argument('-b', '--path_out_b', help='Output path for part-b of the produced split.')
    parser.add_argument('-n', '--num_a', type=int, help='Number of examples that go to split a.')
    parser.add_argument('-r', '--ratio', type=float, help='Between 0 and 1. Amount of data that goes into part-a.')
    parser.add_argument('-l', '--loader', help='The dataset loader to use (in dataset_io).')
    parser.add_argument('-w', '--writer', help='The dataset writer to use (in dataset_io).')
    parser.add_argument('-r', '--random_seed', default=1, type=int, 
                        help='Fix a random seed for reproducability.')
    cmd_args = parser.parse_args()
    random.seed(cmd_args.random_seed)
    main(cmd_args)
