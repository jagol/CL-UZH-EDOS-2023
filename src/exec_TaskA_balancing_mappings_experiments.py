import argparse
import csv
import json
import os
from typing import Any, Dict, List

import delete_checkpoints
import evaluate_predictions
import generate_submission_file
from get_loggers import get_logger
import map_multiclass_to_binary_preds
import predict
import train


RUN_LOGGER = get_logger('run')
EXPERIMENT_NAME = 'balancing_mapping_experiments'
SEEDS = [3, 5, 7]
MODEL_NAME = 'facebook/bart-large'
BALANCING = {
    'TaskA': [None],
    'TaskB': [None, 5, 10],
    'TaskC': [None, 1, 3]
}
NUM_LABELS = {
    'TaskA': 2,
    'TaskB': 5,
    'TaskC': 12
}


def construct_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Construct list of cmd-args/configs."""
    configs = []
    for task in BALANCING:
        for balance in BALANCING[task]:
            if balance is None:
                balancing_str = ''
            else:
                balancing_str = f'_balanced_to_{balance}p'
            fname_train = f'EDOS2023{task}_train_preprocessed{balancing_str}.jsonl'
            trainset_path = os.path.join(args.data_dir, f'EDOS2023{task}', fname_train)

            fname_dev_train_task = f'EDOS2023{task}_dev_preprocessed.jsonl'
            valset_path_train_task = os.path.join(args.data_dir, f'EDOS2023{task}', fname_dev_train_task)

            fname_dev_target_task = f'EDOS2023TaskA_dev_preprocessed.jsonl'
            valset_path_target_task = os.path.join(args.data_dir, f'EDOS2023TaskA', fname_dev_target_task)

            fname_dev_official = 'dev_task_a_entries.csv'
            valset_path_official = os.path.join(args.data_dir, f'EDOS2023TaskA', fname_dev_official)
            
            for seed in SEEDS:
                run_name = f'train_on_{task}_balance_{balance}_seed_{seed}'
                path_base_outdir = os.path.join(args.output_dir, f'{EXPERIMENT_NAME}/{run_name}/')
                configs.append({
                    'train': {
                        # main
                        'experiment_name': EXPERIMENT_NAME,
                        'run_name': run_name,
                        'path_out_dir': path_base_outdir,
                        'model_name': MODEL_NAME,
                        'checkpoint': None,
                        'search_hyperparams': False,
                        'no_cuda': False,
                        # task formulation
                        'task_description': False,
                        'nli': False,
                        'dataset_token': False,
                        'num_labels': NUM_LABELS[task],
                        # data
                        'training_set': trainset_path,
                        'limit_training_set': None,
                        'validation_set': valset_path_train_task,
                        'limit_validation_set': None,
                        # hyperparams
                        'epochs': 10,
                        'patience': 2,
                        'batch_size': 2,
                        'gradient_accumulation': 16,
                        'learning_rate': 1e-5,
                        'warmup_steps': 1000,
                        # reporting and debugging
                        'add_info': False,
                        'log_interval': 10,
                        # evaluation and reporting
                        'evaluation_strategy': 'epoch',
                        'eval_steps': None,
                        # saving
                        'save_strategy': 'epoch',
                        'save_steps': None,
                        # reproducibility
                        'seed': seed,
                        'wandb': True,
                    },
                    'delete_checkpoints': {
                        'path_out_dir': path_base_outdir
                    },
                    # predict on internal and official devset 
                    'predict': {
                        'gpu': 0,
                        'path_out_dir': os.path.join(path_base_outdir, 'predictions'),
                        'eval_set_paths': [valset_path_target_task, valset_path_official],
                        'model_name': MODEL_NAME,
                        'model_checkpoint': path_base_outdir,
                        'hypothesis': None,
                        'dataset_token': False,
                        'task_description': False,
                        'path_strat_config': None
                    },
                    # map and evaluate on interal devset
                    'map_multiclass_to_binary_preds_internal_dev': {
                        'path': os.path.join(path_base_outdir, 'predictions', f'EDOS2023TaskA_dev_preprocessed.jsonl'),
                        'input_key': 'class_probs',
                        'output_key': 'prediction_int_binary'
                    },
                    'evaluate_predictions': {
                        'path_predictions': os.path.join(path_base_outdir, 'predictions', f'EDOS2023TaskA_dev_preprocessed.jsonl'),
                        'evalset_name': f'EDOS2023TaskA',
                        'out_path': os.path.join(path_base_outdir, 'predictions', f'EDOS2023TaskA_dev_preprocessed_metrics.json'),
                        'threshold': 0.5,
                        'pred_key': 'prediction_int_binary',
                        'label_key': 'label_value_binary',
                        'write_false_preds': True
                    },
                    # map official devset and generate submission file
                    'map_multiclass_to_binary_preds_official_dev': {
                        'path': os.path.join(path_base_outdir, 'predictions', 'dev_task_a_entries.jsonl'),
                        'input_key': 'class_probs',
                        'output_key': 'prediction_int_binary'
                    },
                    'generate_submission_file': {
                        'task': 'TaskA',
                        'input': os.path.join(path_base_outdir, 'predictions', 'dev_task_a_entries.jsonl'),
                        'output': os.path.join(path_base_outdir, 'predictions', 'submission_dev_task_a.csv'),
                        'prediction_key': 'prediction_int_binary'
                    }
                })
    return configs


def add_evaluation_to_summary(config: Dict[str, Any], writer: csv.DictWriter) -> None:
    with open(config['evaluate_predictions']['out_path']) as fin:
        metrics = json.load(fin)
        f1_macro = metrics['f1-macro']
        run_name = config['train']['run_name']
        writer.writerow({'f1_macro': f1_macro, 'run_name': run_name})

def exec_run(config: argparse.Namespace, csv_writer: csv.DictWriter) -> None:
    # training
    RUN_LOGGER.info('Start training.')
    train.main(argparse.Namespace(**config['train']))
    RUN_LOGGER.info('Finished training.')
    RUN_LOGGER.info('Delete checkpoints.')
    delete_checkpoints.main(argparse.Namespace(**config['delete_checkpoints']))
    RUN_LOGGER.info('Finished deleting checkpoints.')
    
    # predict on both devsets
    RUN_LOGGER.info('Start prediction on internal dev set.')
    predict.main(argparse.Namespace(**config['predict']))
    RUN_LOGGER.info('Finished prediction on internal dev set.')

    # internal devset
    if config['map_multiclass_to_binary_preds_internal_dev']:
        RUN_LOGGER.info('Start mapping onto binary labels.')
        map_multiclass_to_binary_preds.main(argparse.Namespace(**config['map_multiclass_to_binary_preds_internal_dev']))
        RUN_LOGGER.info('Finished mapping onto binary labels.')
    RUN_LOGGER.info('Evaluate predictions on internal dev set.')
    evaluate_predictions.main(argparse.Namespace(**config['evaluate_predictions']))
    RUN_LOGGER.info('Finished evaluation of predictions on internal dev set.')
    RUN_LOGGER.info('Add evaluation to summary.')
    add_evaluation_to_summary(config, csv_writer)
    
    # official dev-set
    if config['map_multiclass_to_binary_preds_official_dev']:
        RUN_LOGGER.info('Start mapping onto binary labels.')
        map_multiclass_to_binary_preds.main(argparse.Namespace(**config['map_multiclass_to_binary_preds_official_dev']))
        RUN_LOGGER.info('Finished mapping onto binary labels.')
    RUN_LOGGER.info('Generate dev-set submission file.')
    generate_submission_file.main(argparse.Namespace(**config['generate_submission_file']))
    RUN_LOGGER.info('Finished generating dev-set submission file.')


def main(args: argparse.Namespace) -> None:
    list_of_run_configs = construct_configs(args)
    num_runs = len(list_of_run_configs)
    with open(os.path.join(args.output_dir, EXPERIMENT_NAME, 'dev_results_summary.csv'), 'w') as fout:
        fieldnames = ['run_name', 'f1_macro']
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for i, config in enumerate(list_of_run_configs, start=1):
            RUN_LOGGER.info(f'Start run [{i}/{num_runs}].')
            exec_run(config, writer)
            RUN_LOGGER.info(f'Finished run [{i}/{num_runs}].')
            if args.max_runs:
                if i >= args.max_runs:
                    RUN_LOGGER.info('Reached maximum number of runs ({i}). Stopping.')
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', help='Path to the data directory.')
    parser.add_argument('-o', '--output_dir', help='Path to the output directory.')
    parser.add_argument('-m', '--max_runs', required=False, type=int, 
                        help='Limit number of runs executed.')
    cmd_args = parser.parse_args()
    main(cmd_args)
