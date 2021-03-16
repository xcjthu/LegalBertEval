from transformers import BertTokenizer,RobertaTokenizer,AutoTokenizer
import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
import random

class CauseActionFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.max_len = config.getint("train", "max_len")
        self.mode = mode
        if config.get('train', 'PLM_path') == '/data/disk1/private/zhx/bert/ms/':
            self.tokenizer = BertTokenizer.from_pretrained('/data/disk1/private/zhx/bert/ms/vocab.txt')
        else:
            # self.tokenizer = AutoTokenizer.from_pretrained(config.get('train', 'PLM_path'))
            self.tokenizer = AutoTokenizer.from_pretrained(config.get('train', 'PLM_vocab'))

        self.label2id = json.load(open(config.get('data', 'label2id')))

    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        label = []
        for doc in data:
            label.append(self.label2id[doc['label'][0]])
            tokens = self.tokenizer.encode(doc['text'], max_length=self.max_len, add_special_tokens=True, truncation=True)
            mask.append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))
            tokens += [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
            inputx.append(tokens)
        return {
            'input': torch.LongTensor(inputx),
            'mask': torch.LongTensor(mask),
            'label': torch.LongTensor(label),
            'global_mask': torch.LongTensor([[1] + [0] * (self.max_len - 1)] * len(data)),
        }
