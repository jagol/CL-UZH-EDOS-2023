import argparse
import json
import logging
import os
import random
from turtle import TurtleScreenBase
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import sklearn
import torch
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments, Trainer, AutoTokenizer, 
    EarlyStoppingCallback
)
import wandb

from dataset import Dataset
from get_loggers import get_logger


# transformers.logging.set_verbosity_info()
train_logger = get_logger('train')
NLI: bool = False
eval_set: Optional[Dataset] = None
num_comp_metrics_out = 0
main_args: Optional[argparse.Namespace] = None


def filter_inputs(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return {
        'input_ids': batch['input_ids'],
        'attention_mask': batch['attention_mask'],
        'labels': batch['labels']
    }


def map_ternary_labels_to_binary(ternary_labels) -> List[int]:
    """Assumes that 0 references same class in ternary and binary.

    If a label is already binary, it is left unchanged.
    """
    out_labels = []
    for lbl in ternary_labels:
        if lbl == 0:
            out_labels.append(0)
        else:
            out_labels.append(1)
    return out_labels


def train(train_set: Dataset, dev_set: Dataset, model: AutoModelForSequenceClassification,
          tokenizer: AutoTokenizer, args: argparse.Namespace) -> None:
    # for batch in train_set:
    #     break
    # print({k: v.shape for k, v in batch.items()})
    train_logger.info('Instantiate training args.')
    training_args = TrainingArguments(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        output_dir=args.path_out_dir,
        logging_dir=args.path_out_dir,
        logging_steps=args.log_interval,
        save_strategy=args.save_strategy,
        # debugging settings:
        # save_strategy='steps',
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1-macro',
        report_to="wandb" if args.wandb else None,
        no_cuda=args.no_cuda,
    )
    train_logger.info('Instantiate trainer.')
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    train_logger.info('Start training.')
    trainer.train()
    train_logger.info('Finished training.')


def search_hyperparams(train_set: Dataset, dev_set: Dataset, model_init,
                       tokenizer: AutoTokenizer, args: argparse.Namespace) -> None:
    train_logger.info('Instantiate training args.')
    training_args = TrainingArguments(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        output_dir=args.path_out_dir,
        logging_dir=args.path_out_dir,
        logging_steps=args.log_interval,
        save_strategy=args.save_strategy,
        # debugging settings:
        # save_strategy='steps',
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        # report_to="wandb",
    )
    train_logger.info('Instantiate trainer.')
    trainer = Trainer(
        model_init=model_init,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    train_logger.info('Start hyperparameter search.')
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        n_trials=3
    )
    train_logger.info('Finished hyperparameter search.')
    with open(os.path.join(args.path_out_dir, 'hyperparameter_search_results.json')) as fout:
        json.dump(best_run.__dict__, fout)


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    if NLI or max(set(labels.flatten().tolist())) <= 1:
        return compute_binary_metrics(logits, labels)
    else:
        return compute_multi_class_metrics(logits, labels)


def compute_binary_metrics(logits, labels) -> Dict[str, float]:
    if NLI:
        entail_contradiction_logits = torch.FloatTensor(logits[0][:, [0, 2]])
        # predictions = np.argmax(torch.nn.functional.softmax(logits_binary), axis=-1)
        predictions = [int(round(lb[1].item())) for lb in entail_contradiction_logits.softmax(dim=1)]
        labels_bin = map_ternary_labels_to_binary(labels.tolist())
        # predictions = list(entail_contradiction_logits.softmax(dim=1)[:, 1])
    else:
        predictions = [int(p) for p in logits.argmax(axis=1)]
        labels_bin = [label[0] for label in labels.tolist()]

    with open(os.path.join(main_args.path_out_dir, f'comp_metrics_results_{num_comp_metrics_out}.json'), 'w') as fout:
        json.dump({'labels_bin': labels_bin, 'predictions': predictions}, fout)
    return {
        'roc_aux_score': sklearn.metrics.roc_auc_score(labels_bin, predictions),
        'accuracy': sklearn.metrics.accuracy_score(labels_bin, predictions),
        'f1-macro': sklearn.metrics.f1_score(labels_bin, predictions, average='macro'),
        'precision': sklearn.metrics.precision_score(labels_bin, predictions),
        'recall': sklearn.metrics.recall_score(labels_bin, predictions),
    }


def compute_multi_class_metrics(logits, labels) -> Dict[str, float]:
    predictions = [int(p) for p in logits.argmax(axis=1)]
    labels_list = [label[0] for label in labels.tolist()]
    with open(os.path.join(main_args.path_out_dir, f'comp_metrics_results_{num_comp_metrics_out}.json'), 'w') as fout:
        json.dump({'labels': labels_list, 'predictions': predictions}, fout)
    return {
        'accuracy': sklearn.metrics.accuracy_score(labels_list, predictions),
        'f1-macro': sklearn.metrics.f1_score(labels_list, predictions, average='macro'),
        'precision': sklearn.metrics.precision_score(labels_list, predictions, average='macro'),
        'recall': sklearn.metrics.recall_score(labels_list, predictions, average='macro'),
    }


def ensure_valid_encoding(enc_dataset: Dataset) -> None:
    # test correct train-set encoding
    for item in enc_dataset:
        for key, val in item.items():
            if not isinstance(val, torch.Tensor):
                import pdb; pdb.set_trace()


def main(args: argparse.Namespace) -> None:
    global train_logger
    global NLI
    global eval_set
    global main_args
    main_args = args

    if args.nli:
        NLI = True
    if args.wandb:
        project_name = ''
        if 'TaskA' in args.validation_set:
            project_name = 'EDOS2023TaskA'
        elif 'TaskB' in args.validation_set:
            project_name = 'EDOS2023TaskB'
        elif 'TaskC' in args.validation_set:
            project_name = 'EDOS2023TaskC'
        else:
            raise Exception('Validation set "{args.validation_set}" does not point to an EDOS-subtask.')
        wandb.init(project=project_name, entity='jagol', name=args.run_name, tags=[args.experiment_name])

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    path = '/'.join(args.path_out_dir.split('/'))
    if not os.path.exists(path):
        print(f'Output path "{path}" does not exist. Trying to create folder.')
        try:
            os.makedirs(path)
            print(f'Folder "{path}" created successfully.')
        except OSError:
            raise Exception(f'Creation of directory "{path}" failed.')
    if len(os.listdir(path)) > 0:
        raise Exception(f"Output directory '{path}' is not empty.")
        
    train_logger.info('Cmd args: ')
    for k, v in args.__dict__.items():
        train_logger.info(f'"{k}": "{v}"')
    train_logger.info(f'Load tokenizer: {args.model_name}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_logger.info(f'Load trainset from: {args.training_set}')
    train_set = Dataset(name='trainset', path_to_dataset=args.training_set)
    train_set.load(load_limit=args.limit_training_set)
    if args.nli:
        train_set.add_hypotheses('Dummy hypothesis.')
    train_set.encode_dataset(tokenizer=tokenizer, dataset_token=args.dataset_token,
                             task_description=args.task_description)
    ensure_valid_encoding(train_set)

    train_logger.info(f'Load trainset from: {args.validation_set}')
    eval_set = Dataset(name='eval_set', path_to_dataset=args.validation_set)
    eval_set.load(load_limit=args.limit_validation_set)
    if args.nli:
        train_set.add_hypotheses('Dummy hypothesis.')
    eval_set.encode_dataset(tokenizer=tokenizer, dataset_token=args.dataset_token,
                            task_description=args.task_description)

    model_to_load = args.checkpoint if args.checkpoint else args.model_name
    train_logger.info(f'Load Model from: {model_to_load}')

    if args.search_hyperparams:
        train_logger.info(f'Start hyperparameter search with {model_to_load} and num output labels = 2.')

        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(model_to_load, num_labels=2, return_dict=True)
        search_hyperparams(train_set, eval_set, model_init, tokenizer, args)
    else:
        if args.nli:
            train_logger.info(f'Set output layer to dimensionality to {3}')
            model = AutoModelForSequenceClassification.from_pretrained(model_to_load, num_labels=3)
        else:
            train_logger.info(f'Set output layer to dimensionality: {args.num_labels}')
            model = AutoModelForSequenceClassification.from_pretrained(model_to_load, num_labels=args.num_labels)
        train(train_set, eval_set, model, tokenizer, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # main
    parser.add_argument('--experiment_name', required=False, help='Set experiment name. Entered as tag for wandb.')
    parser.add_argument('--run_name', help='Set run name for wandb.')
    parser.add_argument('-o', '--path_out_dir', help='Path to output directory.')
    parser.add_argument('-m', '--model_name',
                        help='Name of model to load. If checkpoint is given this option is still necessary in order to '
                             'load the correct tokenizer.')
    parser.add_argument('-c', '--checkpoint', required=False,
                        help='Optional: provide checkpoint.')
    parser.add_argument('-s', '--search_hyperparams', action='store_true',
                        help='If used, optimize hyperparameters instead of standard training.')
    parser.add_argument('--no_cuda', action='store_true', help='Tell Trainer to not use cuda.')

    # task formulation
    parser.add_argument('--task_description', action='store_true', help='If true, train using task descriptions.')
    parser.add_argument('-n', '--nli', action='store_true', help='If NLI formulation or not.')
    parser.add_argument('--dataset_token', action='store_true', help='If true, add a dataset token to the input.')
    parser.add_argument('-N', '--num_labels', type=int, default=2, help='Only needed if not NLI.')

    # data
    parser.add_argument('-t', '--training_set', help='Path to training data file.')
    parser.add_argument('-L', '--limit_training_set', type=int, default=None,
                        help='Only encode and use <limit_dataset> number of examples.')
    parser.add_argument('-v', '--validation_set', help='Path to development set data file.')
    parser.add_argument('--limit_validation_set', type=int, default=None,
                        help='Only encode and use <limit_dataset> number of examples.')

    # hyperparams
    parser.add_argument('-E', '--epochs', type=float, default=5.0,
                        help='Number of epochs for fine-tuning.')
    parser.add_argument('--patience', type=int, help='Patience for early stopping.')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='Batch-size to be used. Can only be set for training, '
                             'not for inference.')
    parser.add_argument('-a', '--gradient_accumulation', type=int, default=1,
                        help='Number of gradient accumulation steps to perform. The effective '
                             'batch size is batch-size times gradient accumulation steps.')
    parser.add_argument('-l', '--learning_rate', type=float, default=3e-5,
                        help='Learning rate.')
    parser.add_argument('-w', '--warmup_steps', type=int, default=0,
                        help='Number of warmup steps.')

    # reporting and debugging
    parser.add_argument('-A', '--add_info', action='store_true', help='Load additional info into training loop.')
    parser.add_argument('-i', '--log_interval', type=int, default=1,
                        help='Interval batches for which the loss is reported.')

    # evaluation and reporting
    parser.add_argument('--evaluation_strategy', choices=['no', 'steps', 'epoch'], default='epoch')
    parser.add_argument('--eval_steps', type=int, default=None)

    # saving
    parser.add_argument('--save_strategy', default='epoch', choices=['epoch', 'steps', 'no'],
                        help='Analogous to the huggingface-transformers.Trainer-argument.')
    parser.add_argument('--save_steps', default=None, type=int,
                        help='Analogous to the huggingface-transformers.Trainer-argument.')

    # reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for torch, numpy and random. Set for reproducibility.')
    parser.add_argument('--wandb', type=bool, default=True)
    cmd_args = parser.parse_args()
    train_logger: Optional[logging.Logger] = None
    if not cmd_args.wandb:
        # logger = logging.getLogger('wandb')
        # logger.setLevel(logging.WARNING)
        # os.environ['WANDB_DISABLED'] = 'true'
        # os.environ['WANDB_MODE'] = 'disabled'
        wandb.init(mode="disabled")
        # os.environ['WANDB_SILENT'] = 'true'
    main(cmd_args)
