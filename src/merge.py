import argparse
import random

from dataset_io import JSONLLoader, JSONLWriter


random.seed(10)


def main(args: argparse.Namespace) -> None:
    merged_datasets = []
    for path in args.paths_in:
        merged_datasets.extend(JSONLLoader.load(path))
    if args.shuffle:
        random.shuffle(merged_datasets)
    JSONLWriter.write(merged_datasets, args.path_out)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--paths_in', nargs='+', help='Path of input files (jsonl).')
    parser.add_argument('-o', '--path_out', help='Path to output file (jsonl).')
    parser.add_argument('-s', '--shuffle', action='store_true', help='If set, shuffle the merged dataset.')
    cmd_args = parser.parse_args()
    main(cmd_args)
