import argparse
import json
from typing import Dict, Any

from tqdm import tqdm

from get_loggers import get_logger
import split
import preprocess
import merge
import to_label_desc_format


PREPROPIPE_LOGGER = get_logger('prepro-pipeline')


def load_configs(fpath: str) -> Dict[str, Any]:
    with open(fpath) as fin:
        return json.load(fin)


def main(args: argparse.Namespace) -> None:
    PREPROPIPE_LOGGER.info()
    split_configs = load_configs(args.path_split_configs)
    prepro_configs = load_configs(args.path_prepro_configs)
    merge_configs = load_configs(args.path_merge_configs)
    label_desc_configs = load_configs(args.path_label_desc_configs)
    
    PREPROPIPE_LOGGER.info('Split corpora')
    for config_id, split_config in tqdm(split_configs.items()):
        split.main(argparse.ArgumentParser(**split_config))
    
    PREPROPIPE_LOGGER.info('Preprocess corpora')
    for config_id, prepro_config in tqdm(prepro_configs.items()):
        preprocess.main(argparse.ArgumentParser(**prepro_config))
    
    PREPROPIPE_LOGGER.info('Merge corpora')
    for config_id, merge_config in tqdm(merge_configs.items()):
        merge.main(argparse.ArgumentParser(**merge_config))
    
    PREPROPIPE_LOGGER.info('Generate label description versions')
    for config_id, label_desc_config in tqdm(label_desc_configs.items()):
        to_label_desc_format.main(argparse.ArgumentParser(**label_desc_config))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--path_split_configs', help='Path to corpus split configs (json).')
    parser.add_argument('-p', '--path_prepro_configs', help='Path to corpus preprocessing configs (json).')
    parser.add_argument('-m', '--path_merge_configs', help='Path to corpus merging configs (json).')
    parser.add_argument('-d', '--path_label_desc_configs', help='Path to label description configs (json).')
    cmd_args = parser.parse_args()
    main(cmd_args)
