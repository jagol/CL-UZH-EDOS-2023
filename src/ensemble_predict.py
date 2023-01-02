import argparse
from collections import defaultdict
import json
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch

from mappings import CAT_LABEL_NUM_TO_STR, VEC_LABEL_NUM_TO_STR, BIN_LABEL_NUM_TO_STR
from to_label_desc_format import strip_numbering


# *** utility functions ***



def load_preds(fpaths: List[str]) -> List[List[Dict[str, Any]]]:
    preds = []
    for fpath in fpaths:
        model_preds = []
        with open(fpath) as fin:
            for line in fin:
                model_preds.append(json.loads(line))
        preds.append(model_preds)
    return preds


def write_preds_to_outf(preds: List[Dict[str, Any]], fpath: str) -> None:
    with open(fpath, 'w') as fout:
        for pred in preds:
            fout.write(json.dumps(pred) + '\n')


def get_class_probs(item: Dict[str, Any]) -> List[float]:
    if 'prediction' in item:
        return item['prediction']
    elif 'class_probs' in item:
        return item['class_probs']
    else:
        raise Exception(f'Key for class probabilities not found in item: {item}')


# *** same_task_model_pred_averaging ***


def same_task_model_pred_averaging(preds: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Ensemble predictions of multiple models for the same task by averaging their class probabilities."""
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


# *** tasksABC_highest_prob_path ***


def tasksABC_highest_prob_path(preds: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Ensemble predictions by computing the most likely path through the label taxonomy.
    
    Assumes that three predictions files as input, one for each task.
    """
    label_taxonomy = {
        0: {
            0: [0]
        },
        1: {
            1: [1, 2],
            2: [3, 4, 5],
            3: {6, 7, 8, 9},
            4: {10, 11}
        }
    }
    task_preds_ABC = defaultdict(dict)  # {<id>: {"task_A": {...}, "task_B": {...}, "task_C": {...}}}
    ensemble_preds = []
    for task_preds in preds:
        for pred in task_preds:
            num_classes = len(pred['prediction'])
            if num_classes == 2:
                task_preds_ABC[pred['id']]['task_A'] = pred
            elif num_classes == 5:
                task_preds_ABC[pred['id']]['task_B'] = pred
            elif num_classes == 12:
                task_preds_ABC[pred['id']]['task_C'] = pred
            else:
                raise Exception(f'Prediction item has unexpected number of classes: {pred}')
    
    for id_, item in task_preds_ABC.items():
        assert len(item) == 3
    
    for id_ in task_preds_ABC:
        item_A = task_preds_ABC[id_]['task_A']
        item_B = task_preds_ABC[id_]['task_B']
        item_C = task_preds_ABC[id_]['task_C']
        class_probs_A = smooth_distribution(get_class_probs(item_A))
        class_probs_B = smooth_distribution(get_class_probs(item_B))
        class_probs_C = smooth_distribution(get_class_probs(item_C))
        
        path_probs = {}
        for label_A in label_taxonomy:
            for label_B in label_taxonomy[label_A]:
                for label_C in label_taxonomy[label_A][label_B]:
                    path_probs[(label_A, label_B, label_C)] = class_probs_A[label_A] * class_probs_B[label_B] * class_probs_C[label_C]

        task_A_label_probs = [0, 0]
        task_B_label_probs = [0, 0, 0, 0, 0]
        task_C_label_probs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for path in path_probs:
            prob = path_probs[path]
            label_A, label_B, label_C = path
            task_A_label_probs[label_A] += prob
            task_B_label_probs[label_B] += prob
            task_C_label_probs[label_C] += prob
        
        task_A_label_probs = torch.nn.functional.softmax(torch.Tensor(task_A_label_probs), dim=0).tolist()
        task_B_label_probs = torch.nn.functional.softmax(torch.Tensor(task_B_label_probs), dim=0).tolist()
        task_C_label_probs = torch.nn.functional.softmax(torch.Tensor(task_C_label_probs), dim=0).tolist()
        
        item_A['task_A_prediction'] = task_A_label_probs
        item_A['task_B_prediction'] = task_B_label_probs
        item_A['task_C_prediction'] = task_C_label_probs
        item_A['source'] = 'EDOS2023Task_ABC'
        item_A['label_type'] = 'Task_ABC'

        if 'prediction' in item_A:
            del item_A['prediction']
        if 'label_value' in item_A:
            item_A['task_A_label_value'] = item_A['label_value']
            item_A['task_B_label_value'] = item_B['label_value']
            item_A['task_C_label_value'] = item_C['label_value']
            del item_A['label_value']
        if 'label_desc' in item_B:
            item_A['task_A_label_desc'] = item_A['label_desc']
            item_A['task_B_label_desc'] = item_B['label_desc']
            item_A['task_C_label_desc'] = item_C['label_desc']
            del item_A['label_desc']
        ensemble_preds.append(item_A)
    return ensemble_preds


def smooth_distribution(prob_distr: List[float]) -> List[float]:
    # return return torch.nn.functional.softmax(torch.Tensor(prob_distr), dim=0).tolist()
    return prob_distr


# *** main ***


ENSEMBLING_STRATEGIES = {
    'same_task_model_pred_averaging': same_task_model_pred_averaging,
    'tasksABC_highest_prob_path': tasksABC_highest_prob_path,
}


def main(args: argparse.Namespace) -> None:
    preds = load_preds(args.inputs)
    ensembled_preds = ENSEMBLING_STRATEGIES[args.strategy](preds)
    write_preds_to_outf(ensembled_preds, args.output)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', nargs='+', help='List of input files in jsonl containing model predictions with the key class-probs.')
    parser.add_argument('-o', '--output', help='Path to output file. Jsonl format. Contains resulting predictions.')
    parser.add_argument('-s', '--strategy', choices=list(ENSEMBLING_STRATEGIES.keys()), help='Strategy used for ensembling.')
    cmd_args = parser.parse_args()
    main(cmd_args)
