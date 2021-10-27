import enum
import json
import torch
import os
import numpy as np

from transformers import AutoTokenizer,BertTokenizer
from formatter.Basic import BasicFormatter
import random

class LawFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = AutoTokenizer.from_pretrained(config.get('train', 'PLM_vocab'))
        self.max_len = config.getint("train", "max_len")
        self.mode = mode

        self.label2id = json.load(open(config.get("data", "label2id"), "r"))

    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        alllaws = np.zeros((len(data), len(self.label2id)))
        labels = []
        label_mask = np.zeros((len(data), len(self.label2id)))

        for did, temp in enumerate(data):
            tokens = self.tokenizer.encode(temp["inp"], max_length=self.max_len, add_special_tokens=True, truncation=True)
            mask.append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))
            tokens += [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
            inputx.append(tokens)
            selected_l = random.choice(temp["label"])
            labels.append(self.label2id[selected_l])
            for l in temp["label"]:
                if l != selected_l:
                    label_mask[did,self.label2id[l]] = 1
                alllaws[did,self.label2id[l]] = 1

        global_att = np.zeros((len(data), self.max_len), dtype=np.int32)
        global_att[:,0] = 1
        return {
            "text": torch.LongTensor(inputx),
            "mask": torch.LongTensor(mask),
            "global_att": torch.LongTensor(global_att),
            "alllabel": torch.LongTensor(alllaws),
            "label": torch.LongTensor(labels),
            "label_mask": torch.LongTensor(label_mask),
        }
