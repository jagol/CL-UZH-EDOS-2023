import argparse
import json
from typing import *

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import torch

from dataset import Dataset
from get_loggers import get_logger


item_type = Dict[str, Union[float, str, int, Dict[str, Union[int, float]]]]


eval_logger = get_logger('eval_preds')


def to_int(value: float, threshold: float) -> int:
    if value > 1 or value < 0:
        raise Exception('Value needs to be in the interval [0, 1].')
    if value >= threshold:
        return 1
    return 0


def get_true_labels(items: Union[List[item_type], Dataset], label_key: str, threshold: float) -> List[int]:
    labels = [item[label_key] for item in items]
    if isinstance(labels[0], float):
        labels = [to_int(label, threshold=threshold) for label in labels]
    return labels


def get_predictions(items: Union[List[item_type], Dataset], threshold: Optional[float], 
                    pred_key: str) -> Tuple[List[int], List[float]]:
    pred_labels = []
    pred_probs = []
    if isinstance(items[0][pred_key], float) or isinstance(items[0][pred_key], int): 
        for item in items:
            pred_labels.append(to_int(item[pred_key], threshold=threshold))
            pred_probs.append(item[pred_key])
    elif isinstance(items[0][pred_key], list):
        for item in items:
            prob_distr = torch.nn.functional.softmax(torch.Tensor(item[pred_key]), -1)
            largest_index = torch.argmax(prob_distr).item()
            if threshold:
                if prob_distr[largest_index].item() < threshold:
                    pred_labels.append(0)
                    pred_probs.append(None)
                else:
                    pred_labels.append(largest_index)
                    pred_probs.append(prob_distr[largest_index].item())
            else:
                pred_probs.append(prob_distr[largest_index].item())
    return pred_labels, pred_probs


def get_funcs(items: Union[List[item_type], Dataset]) -> List[str]:
    """Get all functionalities that occur in the given list of items."""
    funcs = []
    for item in items:
        if item['functionality'] not in funcs:
            funcs.append(item['functionality'])
    return funcs


def get_items_for_func(items: Union[List[item_type], Dataset], func: str) -> List[item_type]:
    """Get all items of/for the given functionality."""
    items_for_func = []
    for item in items:
        if item['functionality'] == func:
            items_for_func.append(item)
    return items_for_func


def compute_metrics_hatecheck(preds_labels: List[item_type], label_key: str, threshold: float) -> Dict[str, Dict[str, float]]:
    """Compute metrics function for hatecheck.

    Computes evaluation scores for each hatecheck functionality and the overall scores.
    """
    true_labels = get_true_labels(preds_labels, label_key=label_key, threshold=threshold)
    pred_labels, pred_probs = get_predictions(preds_labels, threshold=threshold)

    # compute eval scores overall
    results = {
        'overall': {
            'acc': accuracy_score(true_labels, pred_labels),
            'f1': f1_score(true_labels, pred_labels),
            'recall': recall_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels)
        }
    }

    functionalities = get_funcs(preds_labels)

    # compute functionality-wise scores
    for func in functionalities:
        items_for_func = get_items_for_func(preds_labels, func)
        true_labels_for_func = get_true_labels(items_for_func, label_key=label_key, threshold=threshold)
        pred_labels_for_func, pred_probs_for_func = get_predictions(items_for_func, threshold=threshold)
        assert len(true_labels_for_func) == len(pred_labels_for_func)
        results[func] = {
            'threshold': threshold,
            'acc': accuracy_score(true_labels_for_func, pred_labels_for_func),
            'f1': f1_score(true_labels_for_func, pred_labels_for_func),
            'recall': recall_score(true_labels_for_func, pred_labels_for_func),
            'precision': precision_score(true_labels_for_func, pred_labels_for_func),
            'roc_aux_score': roc_auc_score(true_labels, pred_probs_for_func, average=None),
            'num_examples': len(true_labels_for_func),
            'num_true_hate': sum([i for i in true_labels_for_func]),
            'num_pred_hate': sum([i for i in pred_labels_for_func]),
            'num_true_nohate': len([x for x in true_labels_for_func if x == 0]),
            'num_pred_nohate': len([x for x in pred_labels_for_func if x == 0]),
        }
    return results


def compute_metrics_default(preds_labels: List[item_type], label_key: str, threshold: float, pred_key: str
        ) -> Dict[str, float]:
    """Compute metrics for all datasets except for hatecheck."""
    true_labels = get_true_labels(preds_labels, label_key=label_key, threshold=threshold)
    pred_labels, pred_probs = get_predictions(preds_labels, threshold=threshold, pred_key=pred_key)

    true_class_freqs = {dlabel: true_labels.count(dlabel) for dlabel in set(true_labels)}
    pred_class_freqs = {dlabel: pred_labels.count(dlabel) for dlabel in set(true_labels)}
    num_true_labels = len(set(true_labels))
    num_pred_labels = len(set(pred_labels))

    if num_true_labels != num_pred_labels:
        eval_logger.warning('Gold and predicted labels do not have the same number of distinct labels!')
        eval_logger.warning(f'Gold labels: {set(true_labels)}')
        eval_logger.warning(f'Pred labels: {set(pred_labels)}')
        eval_logger.warning('Using num true labels.')

    return {
        'threshold': threshold,
        'acc': accuracy_score(true_labels, pred_labels),
        'f1-macro': f1_score(true_labels, pred_labels, average='macro'),
        'f1-weighted': f1_score(true_labels, pred_labels, average='weighted'),
        # 'f1-binary': f1_score(true_labels, pred_labels, average='binary'),
        'recall': recall_score(true_labels, pred_labels, average='macro'),
        'precision': precision_score(true_labels, pred_labels, average='macro'),
        # 'roc_aux_score': roc_auc_score(true_labels, pred_probs, average='macro', multi_class='ovr'),
        'num-labels': num_true_labels,
        'true_class_freqs': true_class_freqs,
        'pred_class_freqs': pred_class_freqs
    }


def load_preds_labels(path: str):
    """Load labels from a jsonl-file.

    Returns the entire dicts.
    """
    items = []
    with open(path) as fin:
        for line in fin:
            items.append(json.loads(line))
    return items


def get_false_pos(preds_labels: List[item_type], threshold: float, pred_key: str) -> List[item_type]:
    if isinstance(preds_labels[0][pred_key], list):
        return [item for item in preds_labels if item['label_value'] == 0
            and torch.argmax(torch.Tensor(item[pred_key])).item() == 1]
    return [item for item in preds_labels if item['label_value'] == 0
            and to_int(item[pred_key], threshold) == 1]
    


def get_false_neg(preds_labels: List[item_type], threshold: float, pred_key) -> List[item_type]:
    if isinstance(preds_labels[0][pred_key], list):
        return [item for item in preds_labels if item['label_value'] == 1
            and torch.argmax(torch.Tensor(item[pred_key])).item() == 0]
    return [item for item in preds_labels if item['label_value'] == 1 and 
            to_int(item[pred_key], threshold) == 0]


def main(args: argparse.Namespace) -> None:
    eval_logger.info(f'Load prediction from: {args.path_predictions}')
    preds_labels = load_preds_labels(args.path_predictions)

    if args.evalset_name == 'MHC':
        eval_logger.info('Use method: compute_metrics_hatecheck')
        metrics = compute_metrics_hatecheck(preds_labels, threshold=args.threshold, label_key=args.label_key)
        for metric, val in metrics['overall'].items():
            eval_logger.info(f'{metric}: {val}')
    else:
        eval_logger.info('Use method: compute_metrics_default')
        metrics = compute_metrics_default(preds_labels, threshold=args.threshold, pred_key=args.pred_key, 
                                          label_key=args.label_key)
        for metric, val in metrics.items():
            eval_logger.info(f'{metric}: {val}')

    eval_logger.info(f'Write results to file: {args.out_path}')
    with open(args.out_path, 'w') as fout:
        json.dump(metrics, fout, ensure_ascii=False, indent=4)

    if args.write_false_preds:
        false_pos_items = get_false_pos(preds_labels, threshold=args.threshold, pred_key=args.pred_key)
        false_neg_items = get_false_neg(preds_labels, threshold=args.threshold, pred_key=args.pred_key)

        thresh_repr = f'_{args.threshold}' if args.threshold else ''
        false_pos_path = args.path_predictions[:-6] + f'_false_pos{thresh_repr}.jsonl'
        false_neg_path = args.path_predictions[:-6] + f'_false_neg{thresh_repr}.jsonl'
        
        eval_logger.info(f'Write false positives to: {false_pos_path}')
        with open(false_pos_path, 'w') as fout:
            for item in false_pos_items:
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        eval_logger.info(f'Write false negatives to: {false_neg_path}')
        with open(false_neg_path, 'w') as fout:
            for item in false_neg_items:
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_predictions', help='Path to file containing predictions.')
    parser.add_argument('-d', '--evalset_name', help='Name of dataset.')
    parser.add_argument('-o', '--out_path', help='Path to output file containing scores/metrics.')
    parser.add_argument('-t', '--threshold', type=float, help='For binary predictions: set a '
        'threshold for counting a prediction as class 1.')
    parser.add_argument('-f', '--write_false_preds', action='store_true', 
        help='If true, create files that contain false positives and false negatives.')
    parser.add_argument('-k', '--pred_key', 
        help='Key that points to the prediction that should be evaluated.')
    parser.add_argument('-l', '--label_key', 
                        help='Key that points to the label for the evaluation.')
    cmd_args = parser.parse_args()
    main(cmd_args)
