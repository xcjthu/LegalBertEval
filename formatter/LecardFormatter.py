import json
import torch
import os
import numpy as np

from transformers import AutoTokenizer,BertTokenizer
from formatter.Basic import BasicFormatter

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    # print(len(tokens_a), len(tokens_b))
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class LecardFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = AutoTokenizer.from_pretrained(config.get('train', 'PLM_vocab'))
        self.mode = mode
        self.query_len = config.getint("train", "query_len")
        self.cand_len = config.getint("train", "cand_len")
        self.max_len = self.query_len + self.cand_len + 3

    def process(self, data, config, mode, *args, **params):
        inputx = []
        segment = []
        mask = []
        labels = []

        for temp in data:
            query = self.tokenizer.tokenize(temp["query"])[:self.query_len]
            cand = self.tokenizer.tokenize(temp["cand"])[:self.cand_len]

            tokens = ["[CLS]"] + query + ["[SEP]"] + cand + ["[SEP]"]
            segment_ids = [0] * (len(query) + 2) + [1] * (len(cand) + 1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding = [0] * (self.max_len - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == self.max_len
            assert len(input_mask) == self.max_len
            assert len(segment_ids) == self.max_len

            inputx.append(input_ids)
            segment.append(segment_ids)
            mask.append(input_mask)
            labels.append(int(temp["label"]))

        global_att = np.zeros((len(data), self.max_len), dtype=np.int32)
        global_att[:,0] = 1
        return {
            "inputx": torch.LongTensor(inputx),
            "segment": torch.LongTensor(segment),
            "mask": torch.LongTensor(mask),
            "global_att": torch.LongTensor(global_att),
            "labels": torch.LongTensor(labels),
            "index": [temp["index"] for temp in data]
        }
