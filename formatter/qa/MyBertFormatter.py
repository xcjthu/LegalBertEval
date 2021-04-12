import json
import torch
import numpy as np
import os
from transformers import BertTokenizer


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


class MyBertFormatter:
    def __init__(self, config, mode):
        self.max_len = config.getint("data", "max_len")
        self.multi = config.getboolean("data", "multi_choice")

        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))
        self.k = config.getint("data", "topk")

    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        segment = []
        label = []

        for temp_data in data:
            if self.multi:
                label_x = [0, 0, 0, 0]
                if "A" in temp_data["answer"]:
                    label_x[0] = 1
                if "B" in temp_data["answer"]:
                    label_x[1] = 1
                if "C" in temp_data["answer"]:
                    label_x[2] = 1
                if "D" in temp_data["answer"]:
                    label_x[3] = 1
            else:
                label_x = 0
                if "A" in temp_data["answer"]:
                    label_x = 0
                if "B" in temp_data["answer"]:
                    label_x = 1
                if "C" in temp_data["answer"]:
                    label_x = 2
                if "D" in temp_data["answer"]:
                    label_x = 3
            label.append(label_x)

            inputx.append([])
            mask.append([])
            segment.append([])

            statement = self.tokenizer.tokenize(temp_data["statement"])

            for option in ["A", "B", "C", "D"]:
                ref = []
                k = [0, 1, 2, 6, 12, 7, 13, 3, 8, 9, 14, 15, 4, 10, 11, 5, 16, 17]
                for a in range(0, self.k):
                    res = temp_data["reference"][option][k[a]]
                    ref += self.tokenizer.tokenize(res) + ['[unused1]']
                article = ref[:-1]

                option_tokens = statement + self.tokenizer.tokenize(temp_data["option_list"][option])

                _truncate_seq_pair(article, option_tokens, self.max_len - 3)
                tokens = ["[CLS]"] + article + ["[SEP]"] + option_tokens + ["[SEP]"]
                segment_ids = [0] * (len(article) + 2) + [1] * (len(option_tokens) + 1)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                padding = [0] * (self.max_len - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding

                assert len(input_ids) == self.max_len
                assert len(input_mask) == self.max_len
                assert len(segment_ids) == self.max_len
                inputx[-1].append(input_ids)
                mask[-1].append(input_mask)
                segment[-1].append(segment_ids)
        # print(label)
        return {
            "input": torch.LongTensor(inputx),
            "mask": torch.LongTensor(mask),
            "segment": torch.LongTensor(segment),
            "label": torch.LongTensor(label),
        }
