import argparse
import csv
import json
import os
from typing import Dict, Any

import mappings


def load_preds(fpath: str) -> Dict[str, Any]:
    items = []
    with open(fpath) as fin:
        for line in fin:
            items.append(json.loads(line))
    return items


def main(args: argparse.Namespace) -> None:
    if args.task == 'TaskA':
        mapping = mappings.BIN_LABEL_NUM_TO_STR
    elif args.task == 'TaskB':
        mapping = mappings.CAT_LABEL_NUM_TO_STR
    elif args.task == 'TaskC':
        mapping = mappings.VEC_LABEL_NUM_TO_STR
    else:
        raise Exception(f'Unknown Task: {args.task}')
    # input_fn = args.input.split('/')[-1]
    # input_fn_without_ending = input_fn[:-6]
    preds = load_preds(args.input)
    with open(args.output, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(['rewire_id', 'label_pred'])
        for pred in preds:
            label_str = mapping[pred[args.prediction_key]]
            writer.writerow([pred['id'], label_str])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', choices=['TaskA', 'TaskB', 'TaskC'])
    parser.add_argument('-i', '--input', help='Path to input file in jsonl format.')
    parser.add_argument('-o', '--output', help='Path to output file in csv-format.')
    parser.add_argument('-p', '--prediction_key', 
                        help='Key that holds the prediction for each json-line/item.')
    cmd_args = parser.parse_args()
    main(cmd_args)
