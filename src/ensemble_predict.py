import argparse
from collections import defaultdict
import json
from typing import Any, Dict, List

import numpy as np


def load_preds(fpaths: List[str]) -> List[List[Dict[str, Any]]]:
    preds = []
    for fpath in fpaths:
        model_preds = []
        with open(fpath) as fin:
            for line in fin:
                model_preds.append(json.loads(line))
        preds.append(model_preds)
    return preds


def ensemble_preds(preds: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    id_to_preds = defaultdict(list)
    ensembled_preds = []
    for model_preds in preds:
        for pred in model_preds:
            id_to_preds[pred['id']].append(pred)
    for id, preds in id_to_preds.items():
        list_of_class_probs = [pred['class_probs'] for pred in preds]
        arr = np.array(list_of_class_probs)
        column_average = np.average(arr, axis=0)
        ensemble_pred = dict(preds[0])
        ensemble_pred['class_probs'] = column_average.tolist()
        if 'prediction' in ensemble_pred:
            del ensemble_pred['prediction']
        if 'prediction_int' in ensemble_pred:
            del ensemble_pred['prediction_int']
        if 'prediction_int_binary' in ensemble_pred:
            del ensemble_pred['prediction_int_binary']
        ensembled_preds.append(ensemble_pred)
    return ensembled_preds


def write_preds_to_outf(preds: List[Dict[str, Any]], fpath: str) -> None:
    with open(fpath, 'w') as fout:
        for pred in preds:
            fout.write(json.dumps(pred) + '\n')



def main(args: argparse.Namespace) -> None:
    preds = load_preds(args.inputs)
    ensembled_preds = ensemble_preds(preds)
    write_preds_to_outf(ensembled_preds, args.output)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', nargs='+', help='List of input files in jsonl containing model predictions with the key class-probs.')
    parser.add_argument('-o', '--output', help='Path to output file. Jsonl format. Contains resulting predictions.')
    cmd_args = parser.parse_args()
    main(cmd_args)
