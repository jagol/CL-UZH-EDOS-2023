import argparse
import json
from typing import Dict, Any

from tqdm import tqdm

from get_loggers import get_logger
import join_data_labels
import split
import split_edos
import preprocess
import merge
import to_label_desc_format


PREPROPIPE_LOGGER = get_logger('prepro-pipeline')


def load_configs(fpath: str) -> Dict[str, Any]:
    with open(fpath) as fin:
        return json.load(fin)


def main(args: argparse.Namespace) -> None:
    PREPROPIPE_LOGGER.info('Load configs')
    join_configs = load_configs(args.path_join_configs)
    split_configs = load_configs(args.path_split_configs)
    split_edos_configs = load_configs(args.path_split_edos_configs)
    prepro_configs = load_configs(args.path_prepro_configs)
    merge_configs = load_configs(args.path_merge_configs)
    label_desc_configs = load_configs(args.path_label_desc_configs)
    
    PREPROPIPE_LOGGER.info('Join entries with labels')
    for join_config in tqdm(join_configs):
        join_data_labels.main(argparse.Namespace(**join_config))
    
    PREPROPIPE_LOGGER.info('Split corpora')
    for split_config in tqdm(split_configs):
        split.main(argparse.Namespace(**split_config))
        
    PREPROPIPE_LOGGER.info('Split final EDOS corpora')
    for split_edos_config in tqdm(split_edos_configs):
        split_edos.main(argparse.Namespace(**split_edos_config))
    
    PREPROPIPE_LOGGER.info('Preprocess corpora')
    for prepro_config in tqdm(prepro_configs):
        preprocess.main(argparse.Namespace(**prepro_config))
    
    PREPROPIPE_LOGGER.info('Merge corpora')
    for merge_config in tqdm(merge_configs):
        merge.main(argparse.Namespace(**merge_config))
    
    PREPROPIPE_LOGGER.info('Generate label description versions')
    for label_desc_config in tqdm(label_desc_configs):
        to_label_desc_format.main(argparse.Namespace(**label_desc_config))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--path_join_configs', default=[], help='Path to joining of text and label configs (json).')
    parser.add_argument('-s', '--path_split_configs', default=[], help='Path to corpus split configs (json).')
    parser.add_argument('-e', '--path_split_edos_configs', default=[], help='Path to edos split configs.')
    parser.add_argument('-p', '--path_prepro_configs', default=[], help='Path to corpus preprocessing configs (json).')
    parser.add_argument('-m', '--path_merge_configs', default=[], help='Path to corpus merging configs (json).')
    parser.add_argument('-d', '--path_label_desc_configs', default=[], help='Path to label description configs (json).')
    cmd_args = parser.parse_args()
    main(cmd_args)
