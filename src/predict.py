import json
import os.path
import argparse
import re
from typing import *
# import warnings
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataset import Dataset
from get_loggers import get_logger
from delete_checkpoints import CHECKPOINT_REGEX


# warnings.filterwarnings('ignore')


pred_logger = get_logger('predict')
nli_results_type = Dict[str, Union[float, Dict[str, float]]]


class Predictor:

    def __init__(self, model_name: str, model_checkpoint: Optional[str], device: str) -> None:
        pred_logger.info('Initialize Predictor.')
        self._model_name = model_name
        self._model_checkpoint = model_checkpoint
        self._device = device
        pred_logger.info('Load tokenizer.')
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        pred_logger.info(f'Load model {model_name} from checkpoint {model_checkpoint}.')
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint if model_checkpoint else model_name
        )
        pred_logger.info(f'Move model to device: {self._device}')
        self._model.to(self._device)
        self._model.eval()


class StandardPredictor(Predictor):

    @torch.no_grad()
    def classify(self, input_text: str) -> List[float]:
        """Perform generic classification (no nli), binary or multi-class.

        Args:
            input_text: Text to be classified.
        Return:
            List of class probabilities.
        """
        encoded_input = Dataset.encode_item(self._tokenizer, text=input_text)
        logits = self._model(**encoded_input.to(self._device))[0]
        return torch.softmax(logits.squeeze(), dim=0).tolist()
    
    def classify_w_bin_prob(self, input_text: str) -> float:
        """Classify (binary) and convert to confidence/probability score."""
        encoded_input = Dataset.encode_item(self._tokenizer, text=input_text)
        logits = self._model(**encoded_input.to(self._device))[0]
        return self.logits_to_prob(logits)

    @staticmethod
    def logits_to_prob(logits: torch.FloatTensor):
        prob_distr = torch.softmax(logits.squeeze(), dim=0)
        return float(prob_distr[1])  # take the probability of the positive class


class NLIPredictor(Predictor):

    def __init__(self, model_name: str, model_checkpoint: Optional[str], device: str) -> None:
        # default mapping of Sahajtomar/German_Zeroshot
        self._entail_idx = 0
        self._contra_idx = 2
        super(NLIPredictor, self).__init__(model_name, model_checkpoint, device)

    def classify(self, input_text: str, hypotheses: Union[str, List[str]]) -> Union[float, List[float]]:
        if isinstance(hypotheses, str):
            return self._classify_bin(input_text, hypotheses)
        elif isinstance(hypotheses, list):
            return self._classify_multi(input_text, hypotheses)
        else:
            raise Exception(f"'Hypotheses' should be either str or list. But it is '{type(hypotheses)}'")

    @torch.no_grad()
    def _classify_bin(self, input_text: str, hypothesis: str) -> float:
        """Do binary NLI classification.

        Args:
            input_text: text to be classified/premise
            hypothesis: one hypothesis
        Return:
            prob_entail: The probability of entailment, meaning
                the prob. that the hypothesis is true.
        """
        encoded_input = Dataset.encode_item_with_hypotheses(self._tokenizer, text=input_text, hypothesis=hypothesis)
        del encoded_input['token_type_ids']
        logits = self._model(**encoded_input.to(self._device))[0]
        contradiction_entail_logits = logits[0, [self._contra_idx, self._entail_idx]]  # 0-indexing assumes exactly one
        # example
        probs = contradiction_entail_logits.softmax(dim=0)
        # dim=0 only because already extracted one example with a batch it would be dim=1.
        prob_entail = probs[1].item()
        return prob_entail

    @torch.no_grad()
    def _classify_multi(self, input_text: str, hypotheses: List[str]) -> List[float]:
        """Do NLI classification for more than 2 classes.

        This implies 2 or more hypotheses

        Args:
            input_text: text to be classified/premise
            hypotheses: hypotheses
        Return:
            probs: probabilities corresponding to the hypotheses (same order)
        """
        probs_raw = [self._classify_bin(input_text, hypothesis) for hypothesis in hypotheses]
        # return torch.nn.functional.softmax(torch.FloatTensor(probs_raw), dim=0).tolist()
        return probs_raw


class TaskDescPredictor(Predictor):

    @torch.no_grad()
    def classify(self, input_text: str, task_description: str) -> float:
        """Perform generic classification (no nli), binary or multi-class.

        Args:
            input_text: Text to be classified.
            task_description: describes the task for the classifier.
        Return:
            List of class probabilities.
        """
        encoded_input = Dataset.encode_item_with_task_descriptions(
            self._tokenizer, text=input_text, task_description=task_description)
        del encoded_input['token_type_ids']
        logits = self._model(**encoded_input.to(self._device))[0]
        return self.logits_to_prob(logits)

    @staticmethod
    def logits_to_prob(logits: torch.FloatTensor) -> float:
        prob_distr = torch.softmax(logits.squeeze(), dim=0)
        return float(prob_distr[1])  # take the probability of the positive class


class PredictionPipeline:

    def __init__(self, path_config: str, device: str) -> None:
        self._path_config = path_config
        self._device = device
        self._strategies = {
            'hof_hypos': self._hof_hypos
        }

        with open(path_config) as fin:
            self._config = json.load(fin)
        self._strategy = self._config['strategy']

        # load predictors
        self._predictors = {}  # {'<name>': {'model': model, 'tokenizer': tokenizer}}
        for pred_name in self._config['predictors']:
            model_dict = self._config['predictors'][pred_name]
            self._predictors[pred_name] = {}
            pred_logger.info(f'Load tokenizer and model for {pred_name}')
            if model_dict['type'] == 'NLI':
                self._predictors[pred_name] = NLIPredictor(
                    model_name=model_dict['model'], model_checkpoint=model_dict['checkpoint'], device=device)
            else:
                self._predictors[pred_name] = Predictor(
                    model_name=model_dict['model'], model_checkpoint=model_dict['checkpoint'], device=device)

    def _hof_hypos(self, input_text: str) -> nli_results_type:
        pred_name = list(self._config['predictors'].keys())[0]
        hypotheses = ['Dieser Text enthält Hass.', 'Dieser Text ist obszön.', 'Dieser Text ist beleidigend.']
        probs = self._predictors[pred_name].classify(input_text=input_text, hypotheses=hypotheses)
        return {'final': max(probs), 'individual': {h: p for h, p in zip(hypotheses, probs)}}

    def classify(self, input_text: str) -> nli_results_type:
        return self._strategies[self._strategy](input_text)


def main(args) -> None:
    # create dir if it does not exist
    if not os.path.exists(args.path_out_dir):
        os.mkdir(args.path_out_dir)

    if len(os.listdir(args.path_out_dir)) > 0:
        raise Exception(f'Output directory {args.path_out_dir} is not empty.')

    pred_logger.info('Load test sets:')
    eval_sets = []
    for eval_path in args.eval_set_paths:
        pred_logger.info(f'Load dataset: {eval_path}')
        dataset = Dataset(name='EDOS2023', path_to_dataset=eval_path)
        dataset.load()
        eval_sets.append(dataset)

    device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    # if path is not a model checkpoint, search for a checkpoint in the given directory
    if args.model_checkpoint:
        if not re.fullmatch(CHECKPOINT_REGEX, args.model_checkpoint.split('/')[-1]):
            pred_logger.info(f'"{args.model_checkpoint}" is not a checkpoint. '
                             f'Search for checkpoint in child directories.')
            matches = [1 for name in os.listdir(args.model_checkpoint)
                       if re.fullmatch(CHECKPOINT_REGEX, name) is not None]
            if sum(matches) > 1:
                raise Exception('Multiple checkpoints found.')
            elif sum(matches) < 1:
                raise Exception('No checkpoint found.')
            for name in os.listdir(args.model_checkpoint):
                if re.fullmatch(CHECKPOINT_REGEX, name):
                    args.model_checkpoint = os.path.join(args.model_checkpoint, name)
                    pred_logger.info(f'Found checkpoint "{name}", set checkpoint path to: {args.model_checkpoint}')
    # Load the correct predictor
    if args.path_strat_config:
        predictor = PredictionPipeline(path_config=args.path_strat_config, device=device)
    elif args.hypothesis:
        predictor = NLIPredictor(model_name=args.model_name, model_checkpoint=args.model_checkpoint, device=device)
    elif args.task_description:
        predictor = TaskDescPredictor(model_name=args.model_name, model_checkpoint=args.model_checkpoint, device=device)
    else:
        predictor = StandardPredictor(model_name=args.model_name, model_checkpoint=args.model_checkpoint, device=device)
    pred_logger.info('Start prediction.')
    for eval_set in eval_sets:
        eval_set_path, eval_set_fname = os.path.split(eval_set._path_to_dataset)
        fname, extension = os.path.splitext(eval_set_fname)
        fout = open(os.path.join(args.path_out_dir, fname + '.jsonl'), 'w')
        pred_logger.info(f'Predict on: {eval_set.name}, fname: {os.path.split(eval_set._path_to_dataset)[1]}')
        for item in tqdm(eval_set):
            if args.path_strat_config:
                item['prediction'] = predictor.classify(input_text=item['text'])
            elif args.hypothesis:
                if args.dataset_token:
                    item['prediction'] = predictor.classify(
                        input_text=item['text'], hypothesis=f"[{eval_set.name}] {args.hypothesis}")
                else:
                    item['prediction'] = predictor.classify(input_text=item['text'], hypotheses=args.hypothesis)
            elif args.task_description:
                item['prediction'] = predictor.classify(input_text=item['text'], task_description=item['label_desc'])
            else:
                # do standard classification
                if args.dataset_token:
                    item['class_probs'] = predictor.classify(f"[{item['source']}] {item['text']}")
                else:
                    item['class_probs'] = predictor.classify(input_text=item['text'])
                # item['prediction_int'] = item['class_probs'].index(max(item['class_probs']))
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.close()
    pred_logger.info('Finished prediction.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, help='GPU number to use, -1 means cpu.')
    parser.add_argument('-o', '--path_out_dir', help='Path to an output directory.')
    parser.add_argument('-p', '--eval_set_paths', nargs='+', help='Paths to evaluation sets.')
    # for using a predictor
    parser.add_argument('-m', '--model_name', required=False, help='Hugging-Face name of model/tokenizer.')
    parser.add_argument('-c', '--model_checkpoint', required=False,
                        help='Checkpoint of model to load. Can also be output directory if the output directory only '
                             'contains a single checkpoint directory.')
    parser.add_argument('-H', '--hypothesis', required=False, help='A hypothesis for NLI-based prediction.')
    parser.add_argument('-d', '--dataset_token', action='store_true',
                        help='If dataset token should be used or not. Parent directory name of dataset is used as '
                             'dataset token.')
    parser.add_argument('--task_description', action='store_true', help='If true, use label_type as task description.')
    # for using a prediction pipeline
    parser.add_argument('--path_strat_config', required=False,
                        help='If using a "strategy": use this path to point to the config for the strategy (json).')
    cmd_args = parser.parse_args()
    main(cmd_args)
