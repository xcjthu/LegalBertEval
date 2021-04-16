from transformers import BertTokenizer,RobertaTokenizer,AutoTokenizer
import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
import random

class xsMultiTaskFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.max_len = config.getint("train", "max_len")
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(config.get('train', 'PLM_vocab'))

        self.label2id = json.load(open(config.get('data', 'label2id')))
    
    def generate_label(self, data):
        pass

    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        charge_label = []
        for doc in data:
            tokens = self.tokenizer.encode(doc['text'], max_length=self.max_len, add_special_tokens=True, truncation=True)
            mask.append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))
            tokens += [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
            inputx.append(tokens)
            charge_label.append(self.label2id['charge2id'][doc['charge'][0]])

        return {
            'input': torch.LongTensor(inputx),
            'mask': torch.LongTensor(mask),
            'clabel': torch.LongTensor(charge_label),
            'global_mask': torch.LongTensor([[1] + [0] * (self.max_len - 1)] * len(data)),
        }