import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Union

from compute_corpus_stats import compute_corpus_stats
from get_loggers import get_logger
from mappings import LABEL_STR_TO_LABEL_NUM
from dataset_io import *
from clean import Cleaner

random.seed(42)
prepro_logger = get_logger('preprocessor')


class Preprocessor:
    """Standardize dataset-format and clean text."""
    
    def __init__(self, path_in: str, path_out: str, loader: str, cleaning_type: str, corpus_name: str) -> None:
        self.path_in = path_in
        self.path_out = path_out
        self.loader = LOADERS[loader]
        self.writer = WRITERS['JSONLWriter']
        self.cleaner = Cleaner(cleaning_type=cleaning_type)
        self.corpus_name = corpus_name
    
    def process(self) -> None:
        """The process-method:
            - loads the given input file
            - turn the dataset into the standard dict-format
            - cleans the text
            - writes the resulting dataset to the given output file 
        """
        raise NotImplementedError


class EDOS2023TaskAPreprocessor(Preprocessor):
    
    def process(self) -> None:
        entries = self.loader.load(self.path_in)
        std_entries = []
        for entry in entries:
            std_entry = dict(
                    id=entry['rewire_id'], 
                    text=self.cleaner.clean(entry['text']), 
                    label_type='task_A', 
                    label_value=LABEL_STR_TO_LABEL_NUM[entry['label_sexist']] if 'label_sexist' in entry else None, 
                    source=self.corpus_name,
                )
            std_entries.append(std_entry)
        self.writer.write(std_entries, self.path_out)


class EDOS2023TaskBPreprocessor(Preprocessor):
    
    def process(self) -> None:
        entries = self.loader.load(self.path_in)
        std_entries = []
        for entry in entries:
            std_entry = dict(
                    id=entry['rewire_id'], 
                    text=self.cleaner.clean(entry['text']), 
                    label_type='task_B', 
                    label_value=LABEL_STR_TO_LABEL_NUM[entry['label_category']] if 'label_sexist' in entry else None,
                    source=self.corpus_name,
                )
            std_entries.append(std_entry)
        self.writer.write(std_entries, self.path_out)


class EDOS2023TaskCPreprocessor(Preprocessor):
    
    def process(self) -> None:
        entries = self.loader.load(self.path_in)
        std_entries = []
        for entry in entries:
            std_entry = dict(
                    id=entry['rewire_id'], 
                    text=self.cleaner.clean(entry['text']), 
                    label_type='task_C', 
                    label_value=LABEL_STR_TO_LABEL_NUM[entry['label_vector']] if 'label_sexist' in entry else None,
                    source=self.corpus_name,
                )
            std_entries.append(std_entry)
        self.writer.write(std_entries, self.path_out)


def shuffle_corpus(path: str) -> None:
    with open(path) as fin:
        lines = fin.readlines()
    random.shuffle(lines)
    with open(path, 'w') as fout:
        for line in lines:
            fout.write(line)


def sanity_check(fpath: str) -> None:
    """Perform a sanity check on a given output file of the preprocessor."""
    table = defaultdict(list)
    list_of_dicts = []
    with open(fpath) as fin:
        for line in fin:
            d = json.loads(line)
            list_of_dicts.append(d)
            for key, value in d.items():
                table[key].append(value)

    keys_lenghts = {k: len(v) for k, v in table.items()}
    if len(set(keys_lenghts.values())) != 1:
        raise Exception(f'Sanity check failed: some keys are missing.\n{json.dumps(keys_lenghts)}')

    expected_keys = ['id', 'text', 'split', 'source', 'label_type', 'label_value']
    for d in list_of_dicts:
        for key in d:
            if key not in expected_keys:
                raise Exception(f'Line contained unexpected key: {key}. Line: {d}')


def main(args: argparse.Namespace) -> None:
    processor = CORPUS_TO_Preprocessor[args.corpus](
        args.path_in, 
        args.path_out, 
        args.loader, 
        args.cleaning_type, 
        args.corpus
    )
    processor.process()
    if args.sanity_check:
        sanity_check(args.path_out)
    if args.shuffle:
        shuffle_corpus(args.path_out)
    if args.corpus_stats:
        compute_corpus_stats(args.path_out)


CORPUS_TO_Preprocessor = {
    # 'DGHSD': DGHSDPreprocessor,
    'EDOS2023TaskA': EDOS2023TaskAPreprocessor,
    'EDOS2023TaskB': EDOS2023TaskBPreprocessor,
    'EDOS2023TaskC': EDOS2023TaskCPreprocessor,
    # 'JigsawUBiTC': JigsawUBiTCPreprocessor,
    # 'HateCheck': HateCheckPreprocessor,
    # 'SBF': SBFPreprocessor,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', choices=list(CORPUS_TO_Preprocessor.keys()), help='Name of corpus that is processed.')
    parser.add_argument('-i', '--path_in', help='Path to input file (corpus specific format)')
    parser.add_argument('-o', '--path_out', help='Path to output file (jsonl)')
    parser.add_argument('-l', '--loader', help='Name of loader to use.')
    parser.add_argument('-t', '--cleaning_type', choices=['remove', 'replace'])
    parser.add_argument('--sanity_check', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--corpus_stats', action='store_true')
    cmd_args = parser.parse_args()
    main(cmd_args)
