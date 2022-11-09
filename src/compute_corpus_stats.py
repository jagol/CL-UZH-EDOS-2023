import argparse
import json
from collections import defaultdict
from typing import Any, Dict

from get_loggers import get_logger


stats_logger = get_logger('stats')


def convert_defaultdict_to_dict(d: defaultdict) -> Dict[Any, Any]:
    """Convert arbitrarily nested defaultdict to dict."""
    # from https://stackoverflow.com/questions/26496831/how-to-convert-defaultdict-of-defaultdicts-of-
    # defaultdicts-to-dict-of-dicts-o
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    return d


def compute_percentages(d: Dict[str, int]) -> Dict[str, float]:
    total_num = sum(d.values())
    return {f'{key}_perc': (val / total_num) * 100 for key, val in d.items()}


def compute_corpus_stats(path: str) -> None:
    """Compute corpus-label statistics. Writes results into: <path-without-suffix>.stats.jsonl"""
    global_num_examples = 0
    global_label_counts = defaultdict(lambda: defaultdict(int))  # {label_type: {label_value: count}}
    per_corpus_label_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # {corpus: {label_type: {label_value: count}}}
    per_language_label_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # {language: {label_type: {label_value: count}}}
    corpus_item_count = defaultdict(int)  # {corpus: number of items}

    # go counting :-)
    with open(path) as fin:
        for line in fin:
            d = json.loads(line)
            label_type = d['label_type']
            label_value = d['label_value']
            corpus = d['source']
            global_num_examples += 1
            global_label_counts[label_type][str(label_value)] += 1
            per_corpus_label_counts[corpus][label_type][str(label_value)] += 1
            corpus_item_count[corpus] += 1

    # convert to dict
    global_label_counts = convert_defaultdict_to_dict(global_label_counts)
    per_corpus_label_counts = convert_defaultdict_to_dict(per_corpus_label_counts)
    corpus_item_count = convert_defaultdict_to_dict(corpus_item_count)

    # compute percentages
    for label_type in global_label_counts:
        perc_dict = compute_percentages(global_label_counts[label_type])
        global_label_counts[label_type].update(perc_dict)
    for corpus in per_corpus_label_counts:
        for label_type in per_corpus_label_counts[corpus]:
            perc_dict = compute_percentages(per_corpus_label_counts[corpus][label_type])
            per_corpus_label_counts[corpus][label_type].update(perc_dict)
    corpus_perc_dict = compute_percentages(corpus_item_count)
    corpus_item_count.update(corpus_perc_dict)

    # write to outfile
    path_out = path[:-5] + 'stats.json'
    with open(path_out, 'w') as fout:
        json.dump({
            'global_num_examples': global_num_examples,
            'global_label_counts': global_label_counts,
            'per_corpus_label_counts': per_corpus_label_counts,
            'corpus_item_count': corpus_item_count,
        }, fout, ensure_ascii=False, indent=4)
    stats_logger.info(f'Statistics of resulting dataset written to: {path_out}')


def main(args: argparse.Namespace):
    compute_corpus_stats(args.input)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to input file. Corpus in jsonl-format.')
    cmd_args = parser.parse_args()
    main(cmd_args)
