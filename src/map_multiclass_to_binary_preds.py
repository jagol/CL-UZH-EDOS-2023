import argparse
import csv
import json
import os
from typing import Dict, Any


def load_preds(fpath: str) -> Dict[str, Any]:
    items = []
    with open(fpath) as fin:
        for line in fin:
            items.append(json.loads(line))
    return items


def map_class_probs_to_binary(items: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    for item in items:
        item[args.output_key] = 0 if item[args.input_key].index(max(item[args.input_key])) == 0 else 1
    return items


def map_labels_to_binary(items: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    for item in items:
        item['label_value_binary'] = 0 if item['label_value'] == 0 else 1
    return items


def main(args: argparse.Namespace) -> None:
    preds = load_preds(args.path)
    preds_bin = map_class_probs_to_binary(preds, args)
    if 'label_value' in preds[0]:
        preds_bin = map_labels_to_binary(preds_bin, args)
    with open(args.path, 'w') as fout:
        for pred in preds_bin:
            fout.write(json.dumps(pred) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to input file in jsonl format.')
    parser.add_argument('-i', '--input_key', help='Key to read multiclass value from.')
    parser.add_argument('-o', '--output_key', 'Key to write the binary value to.')
    cmd_args = parser.parse_args()
    main(cmd_args)
