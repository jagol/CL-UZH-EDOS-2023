import argparse
import json
import os
import random
from typing import Optional, Dict, List, Any, Union, NewType
from collections import defaultdict
from statistics import mean, median

import numpy as np
import sklearn
import torch
torch.backends.cuda.matmul.allow_tf32 = True
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments, Trainer, AutoTokenizer, 
    EarlyStoppingCallback, DataCollator,
    AutoModelForSequenceClassification, AutoTokenizer
)
import wandb

from dataclasses import dataclass
from dataset import Dataset
from get_loggers import get_logger
from to_label_desc_format import batch_to_label_desc


# transformers.logging.set_verbosity_info()
TRAIN_LOGGER = None
TOKENIZER = None
MAX_SEQ_LENGTH = 128
DATASET_TOKEN = None
if TRAIN_LOGGER is None:
    TRAIN_LOGGER = get_logger('train')
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

# adjusted from: https://huggingface.co/transformers/v4.8.0/_modules/transformers/data/data_collator.html
InputDataClass = NewType("InputDataClass", Any)
def label_desc_datacollator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """
    texts = [d['input_ids']['text'] for d in features]
    label_types = [d['input_ids']['label_type'] for d in features]
    label_values = [d['input_ids']['label_value'] for d in features]
    sources = [d['input_ids']['source'] for d in features]
    if -1 in label_values:
        import pdb; pdb.set_trace()
    binary_labels, label_descs = batch_to_label_desc(sources, label_types, label_values)
    if DATASET_TOKEN:
        ds_label_descs = []
        for source, label_desc in zip(sources, label_descs):
            ds_label_descs.append(f'[{source}] {label_desc}')
    else:
        ds_label_descs = label_descs
    enc_batch = TOKENIZER(
        text=ds_label_descs, 
        text_pair=texts,
        padding='longest',
        max_length=MAX_SEQ_LENGTH, 
        pad_to_multiple_of=2, 
        return_tensors='pt'
    )
    enc_batch['labels'] = torch.LongTensor(binary_labels)
    return enc_batch


# adjusted from: https://huggingface.co/transformers/v4.8.0/_modules/transformers/data/data_collator.html
InputDataClass = NewType("InputDataClass", Any)
def vanilla_datacollator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """
    texts = [d['input_ids']['text'] for d in features]
    label_values = [d['input_ids']['label_value'] for d in features]
    sources = [d['input_ids']['source'] for d in features]
    if DATASET_TOKEN:
        ds_texts = []
        for source, text in zip(sources, texts):
            ds_texts.append(f'[{source}] {texts}')
    else:
        ds_texts = texts
    enc_batch = TOKENIZER(
        text=ds_texts,
        padding='longest',
        max_length=MAX_SEQ_LENGTH, 
        pad_to_multiple_of=2, 
        return_tensors='pt'
    )
    enc_batch['labels'] = torch.LongTensor(label_values)
    return enc_batch


def train(train_set: Dataset, dev_set: Dataset, model: AutoModelForSequenceClassification,
          tokenizer: AutoTokenizer, args: argparse.Namespace) -> None:
    TRAIN_LOGGER.info('Instantiate training args.')
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
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1-macro',
        report_to='wandb' if args.wandb else None,
        no_cuda=args.no_cuda,
        # tf32=True,
        # bf16=True
    )
    TRAIN_LOGGER.info('Instantiate trainer.')
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=label_desc_datacollator if args.label_desc_datacollator else vanilla_datacollator,
        train_dataset=train_set,
        eval_dataset=dev_set,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    TRAIN_LOGGER.info('Start training.')
    trainer.train()
    TRAIN_LOGGER.info('Finished training.')


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    if max(set(labels.flatten().tolist())) <= 1:
        return compute_binary_metrics(logits, labels)
    else:
        return compute_multi_class_metrics(logits, labels)


def compute_binary_metrics(logits, labels) -> Dict[str, float]:
    predictions = [int(p) for p in logits.argmax(axis=1)]
    labels_bin = [label for label in labels.tolist()]

    with open(os.path.join(main_args.path_out_dir, f'comp_metrics_results_{num_comp_metrics_out}.json'), 'w') as fout:
        json.dump({'labels_bin': labels_bin, 'predictions': predictions}, fout)
    return {
        'roc_auc_score': sklearn.metrics.roc_auc_score(labels_bin, predictions),
        'accuracy': sklearn.metrics.accuracy_score(labels_bin, predictions),
        'f1-macro': sklearn.metrics.f1_score(labels_bin, predictions, average='macro'),
        'precision': sklearn.metrics.precision_score(labels_bin, predictions),
        'recall': sklearn.metrics.recall_score(labels_bin, predictions),
    }


def compute_multi_class_metrics(logits, labels) -> Dict[str, float]:
    predictions = [int(p) for p in logits.argmax(axis=1)]
    labels_list = [label for label in labels.tolist()]
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
    global TRAIN_LOGGER
    if TRAIN_LOGGER is None:
        TRAIN_LOGGER = get_logger('train')
    global eval_set
    global main_args
    global DATASET_TOKEN
    DATASET_TOKEN = args.dataset_token
    main_args = args

    if args.wandb:
        project_name = ''
        if 'TaskA' in args.validation_set:
            project_name = 'EDOS2023TaskA'
        elif 'TaskB' in args.validation_set:
            project_name = 'EDOS2023TaskB'
        elif 'TaskC' in args.validation_set:
            project_name = 'EDOS2023TaskC'
        elif 'EDOS2023' in args.validation_set:
            project_name = 'EDOS2023'
        else:
            raise Exception('Validation set "{args.validation_set}" does not point to EDOS.')
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
        
    TRAIN_LOGGER.info('Cmd args: ')
    for k, v in args.__dict__.items():
        TRAIN_LOGGER.info(f'"{k}": "{v}"')
    TRAIN_LOGGER.info(f'Load tokenizer: {args.model_name}')
    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(args.model_name)

    TRAIN_LOGGER.info(f'Load trainset from: {args.training_set}')
    train_set = Dataset(name='trainset', path_to_dataset=args.training_set)
    train_set.load(load_limit=args.limit_training_set, filter_key=args.filter_key, filter_value=args.filter_value)
    # lengths = []
    # item_num_to_length = {}
    # for i, item in enumerate(train_set):
    #     text = item['input_ids']['text']
    #     label_type = item['input_ids']['label_type']
    #     label_value = item['input_ids']['label_value']
    #     source = item['input_ids']['source']
    #     binary_labels, label_descs = batch_to_label_desc([source], [label_type], [label_value])
    #     if DATASET_TOKEN:
    #         ds_label_descs = []
    #         for s, ld in zip([source], [label_descs]):
    #             ds_label_descs.append(f'[{s}] {ld}')
    #     else:
    #         ds_label_descs = label_descs
    #     enc_batch = TOKENIZER(
    #         text=ds_label_descs[0], 
    #         text_pair=text,
    #         padding=True,
    #         truncation=True,
    #         max_length=MAX_SEQ_LENGTH, 
    #         pad_to_multiple_of=2, 
    #         return_tensors='pt'
    #     )
    #     length = len(enc_batch['input_ids'][0])
    #     lengths.append(length)
    #     item_num_to_length[i] = length
    # length_to_item_num = defaultdict(list)
    # for item_num in item_num_to_length:
    #     length_to_item_num[item_num_to_length[item_num]].append(item_num)
    # sorted_lengths = sorted(lengths)
    # print(f'20 longest sequences: {sorted_lengths[-20:]}')
    # print(f'20 shortest sequences: {sorted_lengths[:20]}')
    # print(f'Average sequence length: {mean(lengths)}')
    # print(f'Median sequence length: {median(lengths)}')

    TRAIN_LOGGER.info(f'Load trainset from: {args.validation_set}')
    eval_set = Dataset(name='eval_set', path_to_dataset=args.validation_set)
    eval_set.load(load_limit=args.limit_validation_set, filter_key=args.filter_key, filter_value=args.filter_value)
    # if not args.label_desc_datacollator:
    #     train_set.encode_dataset(tokenizer=TOKENIZER, dataset_token=args.dataset_token,
    #                             label_description=args.label_description)
    #     # ensure_valid_encoding(train_set)
    #     eval_set.encode_dataset(tokenizer=TOKENIZER, dataset_token=args.dataset_token,
    #                             label_description=args.label_description)

    model_to_load = args.checkpoint if args.checkpoint else args.model_name
    TRAIN_LOGGER.info(f'Load Model from: {model_to_load}')

    TRAIN_LOGGER.info(f'Set output layer to dimensionality: {args.num_labels}')
    model = AutoModelForSequenceClassification.from_pretrained(model_to_load, num_labels=args.num_labels, ignore_mismatched_sizes=True)
    train(train_set, eval_set, model, TOKENIZER, args)


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
    parser.add_argument('--no_cuda', action='store_true', help='Tell Trainer to not use cuda.')

    # task formulation
    parser.add_argument('--label_description', action='store_true', help='If true, train using task descriptions.')
    parser.add_argument('--dataset_token', action='store_true', help='If true, add a dataset token to the input.')
    parser.add_argument('-N', '--num_labels', type=int, default=2)

    # data
    parser.add_argument('-t', '--training_set', help='Path to training data file.')
    parser.add_argument('-L', '--limit_training_set', type=int, default=None,
                        help='Only encode and use <limit_dataset> number of examples.')
    parser.add_argument('-v', '--validation_set', help='Path to development set data file.')
    parser.add_argument('--limit_validation_set', type=int, default=None,
                        help='Only encode and use <limit_dataset> number of examples.')
    parser.add_argument('--filter_key', required=False, help='If set, use this key to filter out instances from the data during loading.')
    parser.add_argument('--filter_value', required=False, help='If set, use this value to filter out instances from the data during loading.')
    parser.add_argument('--label_desc_datacollator', action='store_true', 
                        help='Use the label_desc_datacollator instead of the default datacollator.')

    # hyperparams
    parser.add_argument('-E', '--epochs', type=float, default=5.0,
                        help='Number of epochs for fine-tuning.')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping.')
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
    parser.add_argument('--eval_accumulation_steps', type=int, default=32)

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
    # TRAIN_LOGGER: Optional[logging.Logger] = None
    if not cmd_args.wandb:
        # logger = logging.getLogger('wandb')
        # logger.setLevel(logging.WARNING)
        # os.environ['WANDB_DISABLED'] = 'true'
        # os.environ['WANDB_MODE'] = 'disabled'
        wandb.init(mode="disabled")
        # os.environ['WANDB_SILENT'] = 'true'
    main(cmd_args)
