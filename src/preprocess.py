import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Union

import emoji

from compute_corpus_stats import compute_corpus_stats
from get_loggers import get_logger
from mappings import LABEL_STR_TO_LABEL_NUM


random.seed(42)
prepro_logger = get_logger('preprocessor')


class Preprocessor:
    usr_regex = re.compile(r'@\w+\b')
    url_regex = re.compile("https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:"
                           "[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)")
    white_space_regex = re.compile(r'\s+')
    amp_regex = re.compile(r'&amp;')
    lower_score_regex = re.compile(r'_')
    brackets_regex = re.compile(r'\[.*?\]')

    fn_pattern = '{corpus}_{split}_preprocessed.jsonl'

    def __init__(self, cm_args: argparse.Namespace, outf_main_train, outf_main_dev) -> None:
        self._cmd_args = cm_args
        self._path_corpus_dir = os.path.join(cmd_args.data_dir, self.corpus_name)
        self._outf_main_train = outf_main_train
        self._outf_main_dev = outf_main_dev
        self._outf_train = open(os.path.join(self._path_corpus_dir, self.fn_pattern.format(
            corpus=self.corpus_name, split='train')), 'w')
        self._outf_dev = open(os.path.join(self._path_corpus_dir, self.fn_pattern.format(
            corpus=self.corpus_name, split='dev')), 'w')

    def apply(self) -> None:
        self._process()
        self._outf_train.close()
        self._outf_dev.close()

    def _process(self) -> None:
        """This method handles all processing for a given corpus and should call '_write_to_correct_files'."""
        raise NotImplementedError

    def _write_to_correct_files(self, item: Dict[str, Union[str, int, float]], include_in_main: bool) -> None:
        """Writes an item to the appropriate output files.

        If the item has a different split than 'train' or 'dev' (e.g. 'predict'),
        then do not write to an output file.
        """
        line = json.dumps(item, ensure_ascii=False) + '\n'
        if item['split'] == 'train':
            self._outf_train.write(line)
            if include_in_main:
                self._outf_main_train.write(line)
        elif item['split'] == 'dev':
            self._outf_dev.write(line)
            if include_in_main:
                self._outf_main_dev.write(line)
        elif item['split'] == 'predict':  # For final predictions
            # self._outf_predict_HOF.write(line)
            raise NotImplementedError
        else:
            raise Exception(f'Unexpected split "{item["split"]}", corpus name: "{self.corpus_name}".')

    def clean(self, text: str) -> str:
        ctext = re.sub(self.amp_regex, '&', text)  # sub wrong decoded &amp;
        ctext = re.sub(self.brackets_regex, '', ctext)  # remove stuff in brackets, reserved for special tokens
        if self._cmd_args.processing_type == 'remove':
            ctext = self._clean_remove(ctext)
        elif self._cmd_args.processing_type == 'replace':
            ctext = self._clean_replace(ctext)
        # remove unnecessary white-space
        ctext = re.sub(self.white_space_regex, ' ', ctext)
        return ctext

    def _clean_remove(self, text: str) -> str:
        if self._cmd_args.mentions:
            text = re.sub(self.usr_regex, '', text)
        if self._cmd_args.urls:
            text = re.sub(self.url_regex, '', text)
        if self._cmd_args.emojis:
            text = emoji.replace_emoji(text, replace='')
        return text

    def _clean_replace(self, text: str) -> str:
        if self._cmd_args.mentions:
            text = re.sub(self.usr_regex, '[USR]', text)
        if self._cmd_args.urls:
            text = re.sub(self.url_regex, '[URL]', text)
        if self._cmd_args.emojis:
            text = emoji.demojize(text, language='en', delimiters=('[', ']'))
            text = re.sub(self.lower_score_regex, ' ', text)
        return text

    @staticmethod
    def remove_emojis(text: str) -> str:
        return emoji.get_emoji_regexp().sub(r'', text)

    @staticmethod
    def create_item(id_: Union[str, int], text: str, label_type: str, label_value: int, source: str, 
                    split: Optional[str] = None) -> Dict[str, Union[int, str]]:
        """Create a training/dev item (dict).

        The method standardizes what an item-dict contains.
        "split" is kept as an optional parameter to allow to first create items and later decide on the item's split.
        """
        return {'id': id_, 'text': text, 'label_type': label_type, 'label_value': label_value, 'source': source, 'split': split}


class JigsawUBiTCPreprocessor(Preprocessor):

    def __init__(self, cm_args: argparse.Namespace, outf_main_train, outf_main_dev) -> None:
        self.corpus_name = 'JigsawUBiTC'
        self._jigsaw_size = cmd_args.jigsaw_size
        super(JigsawUBiTCPreprocessor, self).__init__(cm_args, outf_main_train, outf_main_dev)

    def _process(self) -> None:
        path_in_train = os.path.join(self._path_corpus_dir, f'balanced_{self._jigsaw_size}_extract_train.csv')
        path_in_dev = os.path.join(self._path_corpus_dir, 'extract_test.csv')
        for path_in, split in [(path_in_train, 'train'), (path_in_dev, 'dev')]:
            with open(path_in) as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    labels = [
                        ('toxic', int(row['toxicity_bin'])),
                        ('obscene', int(row['obscene_bin'])),
                        ('sexually explicit', int(row['sexual_explicit_bin'])),
                        ('discriminating', int(row['identity_attack_bin'])),
                        ('insulting', int(row['insult_bin'])),
                        ('threatening', int(row['threat_bin'])),
                    ]
                    for label_type, label_value in labels:
                        item = self.create_item(id_=row['id'], text=self.clean(row['comment_text']), label_type=label_type, 
                        label_value=label_value, source=self.corpus_name, split=split)
                        self._write_to_correct_files(item, include_in_main=True)


class EDOS2023Preprocessor(Preprocessor):

    def _process(self) -> None:
        dataset = self._load_dataset(self._fname_raw)
        random.shuffle(dataset)
        for i, item in enumerate(dataset):
            if i < 2000:
                item['split'] = 'dev'
            else:
                item['split'] = 'train'
        
        for item in dataset:
            self._write_to_correct_files(item, include_in_main=True)


class EDOS2023TaskAPreprocessor(EDOS2023Preprocessor):
    
    def __init__(self, cm_args: argparse.Namespace, outf_main_train, outf_main_dev) -> None:
        self.corpus_name = 'EDOS2023TaskA'
        self._fname_raw = 'train_all_tasks.csv'
        super(EDOS2023TaskAPreprocessor, self).__init__(cm_args, outf_main_train, outf_main_dev)
    
    def _load_dataset(self, fname: str) -> List[Dict[str, str]]:
        dataset = []
        with open(os.path.join(self._path_corpus_dir, fname)) as fin:
            dreader = csv.DictReader(fin)
            for row in dreader:
                item = self.create_item(
                    id_=row['rewire_id'], 
                    text=self.clean(row['text']), 
                    label_type='task_A', 
                    label_value=LABEL_STR_TO_LABEL_NUM[row['label_sexist']], 
                    source=self.corpus_name,
                    split=None
                )
                dataset.append(item)
        return dataset


class EDOS2023TaskBPreprocessor(EDOS2023Preprocessor):
    
    def __init__(self, cm_args: argparse.Namespace, outf_main_train, outf_main_dev) -> None:
        self.corpus_name = 'EDOS2023TaskB'
        self._fname_raw = 'train_all_tasks.csv'
        super(EDOS2023TaskBPreprocessor, self).__init__(cm_args, outf_main_train, outf_main_dev)
    
    def _load_dataset(self, fname: str) -> List[Dict[str, str]]:
        dataset = []
        with open(os.path.join(self._path_corpus_dir, fname)) as fin:
            dreader = csv.DictReader(fin)
            for row in dreader:
                item = self.create_item(
                    id_=row['rewire_id'], 
                    text=self.clean(row['text']), 
                    label_type='task_B', 
                    label_value=LABEL_STR_TO_LABEL_NUM[row['label_category']], 
                    source=self.corpus_name,
                    split=None
                )
                dataset.append(item)
        return dataset


class EDOS2023TaskCPreprocessor(EDOS2023Preprocessor):
    
    def __init__(self, cm_args: argparse.Namespace, outf_main_train, outf_main_dev) -> None:
        self.corpus_name = 'EDOS2023TaskC'
        self._fname_raw = 'train_all_tasks.csv'
        super(EDOS2023TaskCPreprocessor, self).__init__(cm_args, outf_main_train, outf_main_dev)
    
    def _load_dataset(self, fname: str) -> List[Dict[str, str]]:
        dataset = []
        with open(os.path.join(self._path_corpus_dir, fname)) as fin:
            dreader = csv.DictReader(fin)
            for row in dreader:
                item = self.create_item(
                    id_=row['rewire_id'], 
                    text=self.clean(row['text']), 
                    label_type='task_C', 
                    label_value=LABEL_STR_TO_LABEL_NUM[row['label_vector']], 
                    source=self.corpus_name,
                    split=None
                )
                dataset.append(item)
        return dataset


class DGHSDPreprocessor(Preprocessor):

    def __init__(self, cm_args: argparse.Namespace, outf_main_train, outf_main_dev) -> None:
        self.corpus_name = 'DGHSD'
        super(DGHSDPreprocessor, self).__init__(cm_args, outf_main_train, outf_main_dev)

    def _process(self) -> None:
        fin = open(os.path.join(self._path_corpus_dir, 'dynamically_generated_hate_speech_dataset_v0.2.3.csv'))
        reader = csv.DictReader(fin)
        for row in reader:
            if row['split'] == 'test':
                continue
            item = self.create_item(id_=row[''], text=self.clean(row['text']), label_type='hate speech',
                                    label_value=1 if row['label'] == 'hate' else 0, source=self.corpus_name, split=row['split'])
            self._write_to_correct_files(item, include_in_main=True)


class SBFPreprocessor(Preprocessor):

    def __init__(self, cm_args: argparse.Namespace, outf_main_train, outf_main_dev) -> None:
        self.corpus_name = 'SBF'
        super(SBFPreprocessor, self).__init__(cm_args, outf_main_train, outf_main_dev)

    def _process(self) -> None:
        path_in_train = os.path.join(self._path_corpus_dir, 'SBIC.v2.agg.trn.csv')
        # path_in_val = os.path.join(self._path_corpus_dir, 'SBIC.v2.agg.dev.csv')
        path_in_dev = os.path.join(self._path_corpus_dir, 'SBIC.v2.agg.tst.csv')
        for path_in, split in [(path_in_train, 'train'), (path_in_dev, 'dev')]:
            with open(path_in) as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    labels = [
                        ('offensive', int(round(float(row['offensiveYN'])))),
                        ('lewd', int(round(float(row['sexYN']))))
                    ]
                    for label_type, label_value in labels:
                        item = self.create_item(id_=row[''], text=self.clean(row['post']), label_type=label_type, 
                            label_value=label_value, source=self.corpus_name, split=split)
                        self._write_to_correct_files(item, include_in_main=True)


class HateCheckPreprocessor(Preprocessor):

    def __init__(self, cm_args: argparse.Namespace, outf_main_train, outf_main_dev) -> None:
        self.corpus_name = 'HateCheck'
        super(HateCheckPreprocessor, self).__init__(cm_args, outf_main_train, outf_main_dev)
        self._fname_raw = 'hatecheck_test.csv'

    def _load_dataset(self, fname: str) -> List[Dict[str, str]]:
        dataset = []
        with open(os.path.join(self._path_corpus_dir, fname)) as fin:
            dreader = csv.DictReader(fin)
            for row in dreader:
                label_num = 1 if row['label_gold'] == 'hateful' else 0
                dataset.append({
                    'id': row[''],
                    'text': row['test_case'],
                    'label_type': row['label_gold'],
                    'target': row['target_ident'],
                    'label_value': label_num,
                    'functionality': row['functionality'],
                    'dataset': self.corpus_name,
                    'split': 'dev',
                })
        return dataset

    def _process(self) -> None:
        dataset = self._load_dataset(self._fname_raw)
        for item in dataset:
            self._write_to_correct_files(item=item, include_in_main=False)


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


def main(args) -> None:
    # create 2 output paths
    fpath_main_train = os.path.join(
        args.out_dir, f'main_train_' + '_'.join(args.corpora) + f'_{args.jigsaw_size}.jsonl')
    fpath_main_dev = os.path.join(
        args.out_dir, f'main_dev_' + '_'.join(args.corpora) + f'_{args.jigsaw_size}.jsonl')
    # open 2 main output files
    outf_main_train = open(fpath_main_train, 'w')
    outf_main_dev = open(fpath_main_dev, 'w')

    # start processing loop
    for corpus in args.corpora:
        if corpus not in CORPUS_TO_Preprocessor:
            raise Exception(f'"{corpus}" is not in CORPUS_TO_Preprocessor.')
        prepro_logger.info(f'Processing: {corpus}')
        prepro = CORPUS_TO_Preprocessor[corpus](cm_args=args, outf_main_train=outf_main_train, outf_main_dev=outf_main_dev)
        prepro.apply()

    outf_main_train.close()
    outf_main_dev.close()

    prepro_logger.info('Do a sanity check on the output.')
    sanity_check(fpath_main_train)
    sanity_check(fpath_main_dev)

    prepro_logger.info('Shuffle the main output corpora.')
    shuffle_corpus(fpath_main_train)
    shuffle_corpus(fpath_main_dev)

    prepro_logger.info('Compute corpus stats main output corpora.')
    compute_corpus_stats(fpath_main_train)
    compute_corpus_stats(fpath_main_dev)

    prepro_logger.info(f'Training set written to: {fpath_main_train}')
    prepro_logger.info(f'Dev set written to: {fpath_main_dev}')


CORPUS_TO_Preprocessor = {
    'DGHSD': DGHSDPreprocessor,
    'EDOS2023TaskA': EDOS2023TaskAPreprocessor,
    'EDOS2023TaskB': EDOS2023TaskBPreprocessor,
    'EDOS2023TaskC': EDOS2023TaskCPreprocessor,
    'JigsawUBiTC': JigsawUBiTCPreprocessor,
    'HateCheck': HateCheckPreprocessor,
    'SBF': SBFPreprocessor,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpora', nargs='+', help='List of names of corpora to be processed.')
    parser.add_argument('-d', '--data_dir', help='Path to data directory.')
    parser.add_argument('-o', '--out_dir', help='Path to output directory.')
    parser.add_argument('-t', '--processing_type', choices=['remove', 'replace', 'None'],
                        help='Determines if specified units (emojis, and/or urls, and/or mentions) should be removed, '
                             'replaced, or leaved unchanged.')
    parser.add_argument('-e', '--emojis', action='store_true', help='Include emoji processing in preprocessing.')
    parser.add_argument('-u', '--urls', action='store_true', help='Include url processing in preprocessing.')
    parser.add_argument('-m', '--mentions', action='store_true',
                        help='Include mention (@) processing in preprocessing.')
    parser.add_argument('-j', '--jigsaw_size', help='Choose the size of jigsaw corpus (actually it is *2).')
    cmd_args = parser.parse_args()
    main(cmd_args)
