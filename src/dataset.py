import csv
import json
import random
from typing import *

import torch
from tqdm import tqdm

from get_loggers import get_logger


random.seed(42)


ds_logger = get_logger('dataset')


def get_numeric_label(item: Dict[str, Union[int, float, str, List[Dict[str, Union[int, float]]]]]) -> torch.LongTensor:
    return torch.LongTensor([int(item['label_value'])])


class Dataset(torch.utils.data.IterableDataset):

    def __init__(self, name: str, path_to_dataset: str) -> None:
        self.name = name
        self._path_to_dataset = path_to_dataset
        self._items = []
        self._hypo_aug_items = []

    def __iter__(self) -> None:
        for item in self._items:
            if 'input_ids' in item:
                try:
                    yield {'input_ids': item['input_ids'], 'token_type_ids': item['token_type_ids'],
                           'attention_mask': item['attention_mask'], 'labels': item['labels']}
                except KeyError:
                    yield {'input_ids': item['input_ids'], 'attention_mask': item['attention_mask'],
                           'labels': item['labels']}
            else:
                yield item

    def __len__(self) -> int:
        return len(self._items)

    def load(self, load_limit: Optional[int] = None) -> None:
        if self._path_to_dataset.endswith('.jsonl'):
            with open(self._path_to_dataset) as fin:
                for i, line in enumerate(fin):
                    d = json.loads(line)
                    self._items.append(d)
                    if load_limit:
                        if i >= load_limit:
                            ds_logger.info(f'Stop dataset loading due to load_limit at {load_limit} items.')
                            break
        elif self._path_to_dataset.endswith('.csv'):
            with open(self._path_to_dataset) as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    row['id'] = row['rewire_id']
                    self._items.append(row)
        

    def add_hypotheses(self, hypothesis: str, augmentation: bool = False) -> None:
        """Add hypotheses and do hypothesis-augmentation."""
        for item in self._items:
            item['hypothesis'] = hypothesis
            if augmentation:
                # new_item = dict(item)
                raise NotImplementedError

    def _has_hypotheses(self) -> bool:
        if 'hypothesis' in self._items[0]:
            return True
        return False

    def encode_dataset(self, tokenizer, dataset_token: bool = False, label_description: bool = False) -> None:
        for item in tqdm(self._items):
            if label_description and dataset_token:
                enc = self.encode_item_with_label_descriptions_and_dataset_token(
                    tokenizer, text=item['text'], source=item['source'], label_description=item['label_desc'])
            elif label_description:
                enc = self.encode_item_with_label_descriptions(
                    tokenizer, text=item['text'], label_description=item['label_desc'])
            elif dataset_token and self._has_hypotheses():
                enc = self.encode_item_with_dataset_token_and_hypotheses(
                    tokenizer, text=item['text'], source=item['source'], hypothesis=item['hypothesis'])
            elif dataset_token:
                enc = self.encode_item_with_dataset_token(tokenizer, text=item['text'], source=item['source'])
            elif self._has_hypotheses():
                enc = self.encode_item_with_hypotheses(tokenizer, text=item['text'], hypothesis=item['hypothesis'])
            else:
                enc = self.encode_item(tokenizer, item['text'])
            enc['labels'] = get_numeric_label(item).squeeze()
            enc['input_ids'] = enc['input_ids'].squeeze()
            try:
                enc['token_type_ids'] = enc['token_type_ids'].squeeze()
            except KeyError:
                pass
            enc['attention_mask'] = enc['attention_mask'].squeeze()
            item.update(enc)

    @staticmethod
    def encode_item(tokenizer, text: str) -> Dict[str, Any]:
        return tokenizer(text=text, return_tensors='pt', truncation=True, padding=True)

    @staticmethod
    def encode_item_with_dataset_token_and_hypotheses(tokenizer, text: str, source: str, hypothesis: str
                                                      ) -> Dict[str, Any]:
        return tokenizer(text=text, text_pair=f"[{source}] {hypothesis}",
                         return_tensors='pt', truncation=True, padding=True, return_token_type_ids=True)

    @staticmethod
    def encode_item_with_dataset_token(tokenizer, text: str, source: str) -> Dict[str, Any]:
        return tokenizer(text=f"[{source}] {text}", return_tensors='pt', truncation=True, padding=True)

    @staticmethod
    def encode_item_with_hypotheses(tokenizer, text: str, hypothesis: str) -> Dict[str, Any]:
        return tokenizer(text=text, text_pair=hypothesis, return_tensors='pt', truncation=True,
                         padding=True, return_token_type_ids=True)

    @staticmethod
    def encode_item_with_label_descriptions(tokenizer, text: str, label_description: str):
        return tokenizer(text=label_description, text_pair=text, return_tensors='pt', truncation=True,
                         padding=True, return_token_type_ids=True)

    @staticmethod
    def encode_item_with_label_descriptions_and_dataset_token(tokenizer, text: str, source: str, label_description: str
                                                             ) -> Dict[str, Any]:
        return tokenizer(text=f'[{source}] {label_description}', text_pair=text, return_tensors='pt',
                         truncation=True, padding=True, return_token_type_ids=True)
