import argparse
import csv
import json
import os
from typing import Dict, Any

from mappings import VECTOR_TO_CAT_MAPPING


def load_preds(fpath: str) -> Dict[str, Any]:
    items = []
    with open(fpath) as fin:
        for line in fin:
            items.append(json.loads(line))
    return items


def map_to_category(items: Dict[str, Any]) -> Dict[str, Any]:
    for item in items:
        predicted_vector = item['prediction'].index(max(item['prediction']))
        item['prediction'] = VECTOR_TO_CAT_MAPPING[predicted_vector]
    return items


def main(args: argparse.Namespace) -> None:
    input_fn = args.input.split('/')[-1]
    input_fn_without_ending = input_fn[:-6]
    preds_multi = load_preds(args.input)
    preds_bin = map_to_category(preds_multi)
    with open(os.path.join(args.output, input_fn_without_ending + '_category.jsonl'), 'w') as fout:
        for pred in preds_bin:
            fout.write(json.dumps(pred) + '\n')
    with open(os.path.join(args.output, input_fn_without_ending + '_category.csv'), 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(['rewire_id', 'label_pred'])
        for pred in preds_bin:
            if pred['prediction'] == 0:
                pred_str = 'not sexist'
            else:
                pred_str = 'sexist'
            writer.writerow([pred['id'], pred_str])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to input file in jsonl format.')
    parser.add_argument('-o', '--output', help='Path to output directory where output files will be written.')
    cmd_args = parser.parse_args()
    main(cmd_args)
