import argparse
import csv
import json
import random
from typing import Any, Dict, List

import compute_corpus_stats
from mappings import BIN_LABEL_NUM_TO_STR, CAT_LABEL_NUM_TO_STR, VEC_LABEL_NUM_TO_STR


random.seed(42)


def strip_numbering(label: str) -> str:
    if '.' in label.split(' ')[0]:
        return ' '.join(label.split(' ')[1:])
    return label


def load_file(fpath: str) -> List[Dict[str, Any]]:
    entries = []
    with open(fpath) as fin:
        if fpath.endswith('.jsonl'):
            for line in fin:
                entries.append(json.loads(line))
        elif fpath.endswith('.csv'):
            reader = csv.DictReader(fin)
            for row in reader:
                entries.append({
                    'id': row['rewire_id'],
                    'text': row['text']
                })
    return entries


def write_dataset_to_file(dataset: List[Dict[str, Any]], fpath: str) -> None:
    with open(fpath, 'w') as fout:
        for item in dataset:
            fout.write(json.dumps(item) + '\n')


def to_label_desc_format(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dataset_label_desc = []
    for item in dataset:
        desc_item = dict(item)
        if item['label_type'] == 'task_A':
            desc_item['label_desc'] = 'sexist'
        elif item['label_type'] == 'task_B':
            if item['label_value'] == 0:
                label = random.randint(1, 4)  # choose random label (not None/0)
                desc_item['label_desc'] = f'{strip_numbering(CAT_LABEL_NUM_TO_STR[label])} (against women)'
                desc_item['orig_label_value'] = desc_item['label_value']
                desc_item['label_value'] = 0
            else:
                desc_item['label_desc'] = f"{strip_numbering(CAT_LABEL_NUM_TO_STR[item['label_value']])} (against women)"
                desc_item['orig_label_value'] = desc_item['label_value']
                desc_item['label_value'] = 1
        elif item['label_type'] == 'task_C':
            if item['label_value'] == 0:
                label = random.randint(1, 11)  # choose random label (not None/0)
                desc_item['label_desc'] = f'{strip_numbering(VEC_LABEL_NUM_TO_STR[label])} (against women)'
                desc_item['orig_label_value'] = desc_item['label_value']
                desc_item['label_value'] = 0
            else:
                desc_item['label_desc'] = f"{strip_numbering(VEC_LABEL_NUM_TO_STR[item['label_value']])} (against women)"
                desc_item['orig_label_value'] = desc_item['label_value']
                desc_item['label_value'] = 1
        elif item['label_type'] == 'hate speech':
            desc_item['label_desc'] = 'hate speech'
        elif item['label_type'] == 'offensive':
            desc_item['label_desc'] = 'offensive'
        elif item['label_type'] == 'lewd':
            desc_item['label_desc'] = 'lewd'
        else:
            raise Exception(f"Unexpected label type: {item['label_type']}")
        dataset_label_desc.append(desc_item)
    return dataset_label_desc


def add_binary_label_desc(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    new_dataset = []
    for item in dataset:
        if item['source'] == 'EDOS2023TaskA':
            item['label_desc'] = 'sexist'
        elif item['source'] == 'DGHSD':
            item['label_desc'] = 'hate speech'
        new_dataset.append(item)
    return new_dataset


def main(args: argparse.Namespace) -> None:
    dataset = load_file(args.input)
    if args.binary:
        dataset = add_binary_label_desc(dataset)
        write_dataset_to_file(dataset, args.output)
    else:
        dataset = to_label_desc_format(dataset)
        write_dataset_to_file(dataset, args.output)
        compute_corpus_stats.compute_corpus_stats(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to input file, jsonl- or csv-format.')
    parser.add_argument('-o', '--output', help='Path to output file, jsonl-label-desc-format.')
    parser.add_argument('-b', '--binary', action='store_true', 
                        help='Add binary label descriptions. No label values required.')
    cmd_args = parser.parse_args()
    main(cmd_args)
