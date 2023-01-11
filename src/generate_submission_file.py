import argparse
import csv
import json
import os
from typing import Dict, Any
import zipfile

import torch

from evaluate_predictions import to_int
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
    preds = load_preds(args.input)
    with open(args.output, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(['rewire_id', 'label_pred'])
        for pred in preds:
            if args.to_int:
                if isinstance(pred[args.prediction_key], float):
                    pred_int = to_int(pred[args.prediction_key], args.threshold)
                elif isinstance(pred[args.prediction_key], list):
                    if isinstance(pred[args.prediction_key][1], list):
                        bin_pred, fine_grained_preds = pred[args.prediction_key]
                        if bin_pred > args.threshold:
                            pred_int = torch.argmax(torch.FloatTensor(fine_grained_preds)).item() + 1
                        else:
                            pred_int = 0
                    else:
                        pred_int = torch.argmax(torch.Tensor(pred[args.prediction_key]), dim=-1).item()
            else:
                pred_int = pred[args.prediction_key]
            label_str = mapping[pred_int]
            writer.writerow([pred['id'], label_str])
    fname = os.path.split(args.output)[1]
    with zipfile.ZipFile(args.path_zip, mode='w') as archive:
        archive.write(args.output, arcname=fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', choices=['TaskA', 'TaskB', 'TaskC'])
    parser.add_argument('-i', '--input', help='Path to input file in jsonl format.')
    parser.add_argument('-o', '--output', help='Path to output file in csv-format.')
    parser.add_argument('-z', '--path_zip', help='Path to zip-archive continaing the resulting csv-file.')
    parser.add_argument('-p', '--prediction_key', 
                        help='Key that holds the prediction for each json-line/item.')
    parser.add_argument('--to_int', action='store_true', 
                        help='If predictions are floats, use this flag to convert the floats (between 0 and 1] to an int.')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Threshold used when converting a float to an integer.')
    cmd_args = parser.parse_args()
    main(cmd_args)
