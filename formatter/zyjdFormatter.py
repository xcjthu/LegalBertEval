import enum
import json
import torch
import os
import numpy as np

from transformers import AutoTokenizer,BertTokenizer
from formatter.Basic import BasicFormatter
import random

class zyjdFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = AutoTokenizer.from_pretrained(config.get('train', 'PLM_vocab'))
        self.max_len = config.getint("train", "max_len") + 10
        self.mode = mode
        self.read_label(config)

    def read_label(self, config):
        label2num = json.load(open(config.get("data", "label2num")))
        num_threshold = config.getint("data", "threshold")
        self.label2id = {"NA": 0}
        for l in label2num:
            if label2num[l] > num_threshold:
                # key = "/".join(l.split("/")[:2])
                # key = l.split("/")[0]
                key = l
                if key not in self.label2id:
                    self.label2id[key] = len(self.label2id)

    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        labels = []
        # label_mask = np.zeros((len(data), len(self.label2id)))

        for did, temp in enumerate(data):
            tokens = [self.tokenizer.cls_token_id] * 10 + self.tokenizer.encode(temp["sent"], max_length=self.max_len - 10, add_special_tokens=True, truncation=True)
            mask.append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))
            tokens += [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
            inputx.append(tokens)
            # labels.append(self.label2id["/".join(random.choice(temp["label"]).split("/")[:2])])
            labels.append(self.label2id[random.choice(temp["label"])])

        return {
            "text": torch.LongTensor(inputx),
            "mask": torch.LongTensor(mask),
            "label": torch.LongTensor(labels),
            "ids": [t["id"] for t in data],
        }
