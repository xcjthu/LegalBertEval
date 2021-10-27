import enum
import json
import torch
import os
import numpy as np

from transformers import AutoTokenizer,BertTokenizer
from formatter.Basic import BasicFormatter
import random

class CNNLawFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = json.load(open(config.get("data", "word2id"), "r", encoding="utf8"))
        self.max_len = config.getint("train", "max_len")
        self.mode = mode

        self.label2id = json.load(open(config.get("data", "label2id"), "r"))

    def convert_tokens_to_ids(self, text):
        arr = []
        for a in range(0, len(text)):
            if text[a] in self.tokenizer.keys():
                arr.append(self.tokenizer[text[a]])
            else:
                arr.append(self.tokenizer["[UNK]"])
        return arr[:self.max_len]

    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        alllaws = np.zeros((len(data), len(self.label2id)))
        labels = []
        label_mask = np.zeros((len(data), len(self.label2id)))

        for did, temp in enumerate(data):
            tokens = self.convert_tokens_to_ids(temp["inp"])# self.tokenizer.encode(temp["inp"], max_length=self.max_len, add_special_tokens=True, truncation=True)
            mask.append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))
            tokens += [self.tokenizer["[PAD]"]] * (self.max_len - len(tokens))
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
            "label": torch.LongTensor(alllaws),
            # "label": torch.LongTensor(labels),
            # "label_mask": torch.LongTensor(label_mask),
        }
