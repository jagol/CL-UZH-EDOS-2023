import json
import os.path
import argparse
import re
from typing import *
# import warnings
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DebertaV2ForSequenceClassification, DebertaV2Tokenizer

from dataset import Dataset
from get_loggers import get_logger
from delete_checkpoints import CHECKPOINT_REGEX
from mappings import CAT_LABEL_NUM_TO_STR, VEC_LABEL_NUM_TO_STR, LABEL_STR_TO_LABEL_NUM, BIN_LABEL_NUM_TO_STR
from to_label_desc_format import strip_numbering


# warnings.filterwarnings('ignore')


pred_logger = get_logger('predict')
nli_results_type = Dict[str, Union[float, Dict[str, float]]]


class Predictor:

    def __init__(self, model_name: str, model_checkpoint: Optional[str], device: str, dataset_token: Optional[bool] = None) -> None:
        pred_logger.info('Initialize Predictor.')
        self._model_name = model_name
        self._model_checkpoint = model_checkpoint
        self._device = device
        self._dataset_token = dataset_token
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
        if self._dataset_token:
            encoded_input = Dataset.encode_item_with_dataset_token(
                self._tokenizer, text=input_text, source=self._dataset_token)
        else:
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


class StaggeredPredictor(Predictor):
    
    def __init__(self, model_name_1: str, model_checkpoint_1: Optional[str], dataset_token_1: str, 
                 model_name_2: str, model_checkpoint_2: str, dataset_token_2: str, device: str) -> None:
        pred_logger.info('Initialize Predictor.')
        self._model_name_1 = model_name_1
        self._model_checkpoint_1 = model_checkpoint_1
        self._dataset_token_1 = dataset_token_1
        self._model_name_2 = model_name_2
        self._model_checkpoint_2 = model_checkpoint_2
        self._dataset_token_2 = dataset_token_2
        self._device = device
        # model 1: binary decision
        pred_logger.info('Load tokenizer.')
        self._tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)
        pred_logger.info(f'Load model {model_name_1} from checkpoint {model_checkpoint_1}.')
        self._model_1 = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint_1 if model_checkpoint_1 else model_name_1
        )
        pred_logger.info(f'Move model to device: {self._device}')
        self._model_1.to(self._device)
        self._model_1.eval()
        # model 2: fine-grained decision
        pred_logger.info('Load tokenizer.')
        self._tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)
        pred_logger.info(f'Load model {model_name_1} from checkpoint {model_checkpoint_2}.')
        self._model_2 = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint_2 if model_checkpoint_2 else model_name_2
        )
        pred_logger.info(f'Move model to device: {self._device}')
        self._model_2.to(self._device)
        self._model_2.eval()
    
    @torch.no_grad()
    def classify(self, input_text: str) -> Tuple[float, List[float]]:
        raise NotImplementedError


class StaggeredStandardPredictor(StaggeredPredictor):
    
    @torch.no_grad()
    def classify(self, input_text: str) -> Tuple[float, List[float]]:
        """Perform generic classification (no nli), binary or multi-class.

        Args:
            input_text: Text to be classified.
        Return:
            List of class probabilities.
        """
        if self._dataset_token_1:
            encoded_input_1 = Dataset.encode_item_with_dataset_token(
                self._tokenizer_1, text=input_text, source=self._dataset_token_1)
        else:
            encoded_input_1 = Dataset.encode_item(self._tokenizer_1, text=input_text)
        if self._dataset_token_2:
            encoded_input_2 = Dataset.encode_item_with_dataset_token(
                self._tokenizer_2, text=input_text, source=self._dataset_token_2)
        else:
            encoded_input_2 = Dataset.encode_item(self._tokenizer_2, text=input_text)
        encoded_input_2 = Dataset.encode_item(self._tokenizer_2, text=input_text)
        bin_pred_logits = self._model_1(**encoded_input_1.to(self._device))[0].squeeze()
        bin_pred_prob = torch.softmax(bin_pred_logits, dim=0)[1]
        fine_grained_logits = self._model_2(**encoded_input_2.to(self._device))[0]
        return float(bin_pred_prob.item()), torch.softmax(fine_grained_logits.squeeze(), dim=0).tolist()


class StaggeredLabelDescStandardPredictor(StaggeredPredictor):
    
    @torch.no_grad()
    def classify(self, input_text: str, label_description: str) -> Tuple[float, List[float]]:
        """Perform generic classification (no nli), binary or multi-class.

        Args:
            input_text: Text to be classified.
        Return:
            List of class probabilities.
        """
        if self._dataset_token_1:
            encoded_input_1 = Dataset.encode_item_with_label_descriptions_and_dataset_token(
                self._tokenizer_1, text=input_text, source=self._dataset_token_1, label_description=label_description)
        else:
            encoded_input_1 = Dataset.ncode_item_with_label_descriptions(
                self._tokenizer_1, text=input_text, label_description=label_description)
        if self._dataset_token_2:
            encoded_input_2 = Dataset.encode_item_with_dataset_token(
                self._tokenizer_2, text=input_text, source=self._dataset_token_2)
        else:
            encoded_input_2 = Dataset.encode_item(self._tokenizer_2, text=input_text)
        encoded_input_1 = Dataset.encode_item_with_label_descriptions(self._tokenizer_1, text=input_text)
        encoded_input_2 = Dataset.encode_item(self._tokenizer_2, text=input_text)
        bin_pred_logits = self._model_1(**encoded_input_1.to(self._device))[0].squeeze()
        bin_pred_prob = torch.softmax(bin_pred_logits, dim=0)[1]
        fine_grained_logits = self._model_2(**encoded_input_2.to(self._device))[0]
        return float(bin_pred_prob.item()), torch.softmax(fine_grained_logits.squeeze(), dim=0).tolist()


class TaskDescPredictor(Predictor):

    def classify(self, input_text: str, label_description: str) -> float:
        """Perform generic classification (no nli), binary or multi-class.

        Args:
            input_text: Text to be classified.
            label_description: describes the task for the classifier.
        Return:
            Probabilities of the positive class.
        """
        return self._classify(input_text, label_description)

    @torch.no_grad()
    def _classify(self, input_text: str, label_description: str) -> float:
        """Perform generic classification (no nli), binary or multi-class.

        Args:
            input_text: Text to be classified.
            label_description: describes the task for the classifier.
        Return:
            Probabilities of the positive class.
        """
        if self._dataset_token:
            encoded_input = Dataset.encode_item_with_label_descriptions_and_dataset_token(
                self._tokenizer, text=input_text, source=self._dataset_token , label_description=label_description)
        else:
            encoded_input = Dataset.encode_item_with_label_descriptions(
                self._tokenizer, text=input_text, label_description=label_description)
        # del encoded_input['token_type_ids']
        logits = self._model(**encoded_input.to(self._device))[0]
        return self.logits_to_prob(logits)

    @staticmethod
    def logits_to_prob(logits: torch.FloatTensor) -> float:
        prob_distr = torch.softmax(logits.squeeze(), dim=0)
        return float(prob_distr[1])  # take the probability of the positive class


class TaskDescCategoryPredictor(TaskDescPredictor):
    
    def classify(self, input_text: str, label_description: Optional[str] = None) -> List[float]:
        """Perform generic classification (no nli), binary or multi-class.

        Args:
            input_text: Text to be classified.
            label_description: describes the task for the classifier.
        Return:
            Probabilities of the positive class.
        """
        cat_probs = []
        for label, cat_str in CAT_LABEL_NUM_TO_STR.items():
            cat_probs.append(self._classify(input_text, strip_numbering(cat_str)))
        return cat_probs


class TaskDescVectorPredictor(TaskDescPredictor):
    
    def classify(self, input_text: str, label_description: Optional[str] = None) -> List[float]:
        """Perform generic classification (no nli), binary or multi-class.

        Args:
            input_text: Text to be classified.
            label_description: describes the task for the classifier.
        Return:
            Probabilities of the positive class.
        """
        cat_probs = []
        for label, vec_str in VEC_LABEL_NUM_TO_STR.items():
            cat_probs.append(self._classify(input_text, strip_numbering(vec_str)))
        return cat_probs
    

class TaskDescVectorPredictorMaxToBin(TaskDescPredictor):
    
    def classify(self, label_description: str, input_text: str) -> List[float]:
        """Perform generic classification (no nli), binary or multi-class.

        Args:
            input_text: Text to be classified.
            label_description: describes the task for the classifier.
        Return:
            Probabilities of the positive class.
        """
        cat_probs = []
        for label, vec_str in VEC_LABEL_NUM_TO_STR.items():
            cat_probs.append(self._classify(input_text, strip_numbering(vec_str)))
        return max(cat_probs)


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


def get_model_checkpoint_path(model_checkpoint: Optional[str]) -> str:
    """If path is not a model checkpoint, search for a checkpoint in the given directory."""
    if model_checkpoint:
        if not re.fullmatch(CHECKPOINT_REGEX, model_checkpoint.split('/')[-1]):
            pred_logger.info(f'"{model_checkpoint}" is not a checkpoint. '
                             f'Search for checkpoint in child directories.')
            matches = [1 for name in os.listdir(model_checkpoint)
                       if re.fullmatch(CHECKPOINT_REGEX, name) is not None]
            if sum(matches) > 1:
                raise Exception('Multiple checkpoints found.')
            elif sum(matches) < 1:
                raise Exception('No checkpoint found.')
            for name in os.listdir(model_checkpoint):
                if re.fullmatch(CHECKPOINT_REGEX, name):
                    model_checkpoint = os.path.join(model_checkpoint, name)
                    pred_logger.info(f'Found checkpoint "{name}", set checkpoint path to: {model_checkpoint}')
    return model_checkpoint


def get_predictor(args: argparse.Namespace, model_checkpoint: str, device: str) -> Predictor:
    if args.path_strat_config:
        predictor = PredictionPipeline(path_config=args.path_strat_config, device=device)
    elif args.predictor == 'StaggeredStandardPredictor':
        model_checkpoint = get_model_checkpoint_path(args.model_checkpoint)
        model_checkpoint_2 = get_model_checkpoint_path(args.model_checkpoint_2)
        predictor = StaggeredStandardPredictor(model_name_1=args.model_name, model_checkpoint_1=model_checkpoint, dataset_token=args.dataset_token,
                                               model_name_2=args.model_name_2, model_checkpoint_2=model_checkpoint_2, dataset_token_2=args.dataset_token_2,
                                               device=device)
    else:
        model_checkpoint = get_model_checkpoint_path(args.model_checkpoint)
        predictor = PREDICTORS[args.predictor](model_name=args.model_name, model_checkpoint=model_checkpoint, 
                                               device=device, dataset_token=args.dataset_token)
    return predictor


PREDICTORS = {
    'PredictionPipeline': PredictionPipeline,
    'TaskDescPredictor': TaskDescPredictor,
    'TaskDescCategoryPredictor': TaskDescCategoryPredictor,
    'TaskDescVectorPredictorMaxToBin': TaskDescVectorPredictorMaxToBin,
    'TaskDescVectorPredictor': TaskDescVectorPredictor,
    'StaggeredStandardPredictor': StaggeredStandardPredictor,
    'StandardPredictor': StandardPredictor
}


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
    predictor = get_predictor(args, device)
    # Load the correct predictor
    
    pred_logger.info('Start prediction.')
    for i, eval_set in enumerate(eval_sets):
        eval_set_path, eval_set_fname = os.path.split(eval_set._path_to_dataset)
        if args.fnames_out:
            fname = args.fnames_out[i]
        else:
            fname, extension = os.path.splitext(eval_set_fname)
            fname += '.jsonl'
        fout = open(os.path.join(args.path_out_dir, fname), 'w')
        pred_logger.info(f'Predict on: {eval_set.name}, fname: {os.path.split(eval_set._path_to_dataset)[1]}')
        for item in tqdm(eval_set):
            if args.path_strat_config:
                item['prediction'] = predictor.classify(input_text=item['text'])
            elif args.label_description and args.predictor == 'TaskDescPredictor':
                # Only pass label_desc argument for the binary prediction done by TaskDescPredictor.
                # Other Multi-label predictors that the label descriptions already saved internally.
                item['prediction'] = predictor.classify(input_text=item['text'], label_description=item['label_desc'])
            else:
                # do standard classification
                item['prediction'] = predictor.classify(input_text=item['text'])
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.close()
    pred_logger.info('Finished prediction.')    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, help='GPU number to use, -1 means cpu.')
    parser.add_argument('-o', '--path_out_dir', help='Path to an output directory.')
    parser.add_argument('-p', '--eval_set_paths', nargs='+', help='Paths to evaluation sets.')
    parser.add_argument('--fnames_out', nargs='+', required=False, help='Names of outfiles. Sorting corresponds to evaluation set-paths. '
                        'If not set, filename is constructed automatically from the evaluation set.')
    # for using a predictor
    parser.add_argument('-m', '--model_name', required=False, help='Hugging-Face name of model/tokenizer.')
    parser.add_argument('-c', '--model_checkpoint', required=False,
                        help='Checkpoint of model to load. Can also be output directory if the output directory only '
                             'contains a single checkpoint directory.')
    parser.add_argument('--model_name_2', required=False, help='Second model name if multiple models are used for prediction')
    parser.add_argument('--model_checkpoint_2', required=False, help='Second model name if multiple checkpoints are used for prediction')
    parser.add_argument('-H', '--hypothesis', required=False, help='A hypothesis for NLI-based prediction.')
    parser.add_argument('-d', '--dataset_token', action='store_true',
                        help='If dataset token should be used or not. Parent directory name of dataset is used as '
                             'dataset token.')
    parser.add_argument('--label_description', action='store_true', help='If true, use label_type as task description.')
    parser.add_argument('--label_desc_category', action='store_true', help='Predict sexism categories')
    # for using a prediction pipeline
    parser.add_argument('--path_strat_config', required=False,
                        help='If using a "strategy": use this path to point to the config for the strategy (json).')
    parser.add_argument('--predictor', choices=list(PREDICTORS.keys()))
    cmd_args = parser.parse_args()
    main(cmd_args)
