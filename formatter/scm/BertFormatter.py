import json
import torch
import os
import numpy as np

from transformers import AutoTokenizer

from formatter.Basic import BasicFormatter


class BertSCM(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = AutoTokenizer.from_pretrained(config.get('train', 'PLM_vocab'))
        self.max_len = config.getint("train", "max_len")
        self.mode = mode

    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        labels = []

        for temp in data:
            inputx.append([])
            mask.append([])
            for name in ["A", "B", "C"]:
                text = temp[name]
                tokens = self.tokenizer.encode(text, max_length=self.max_len, add_special_tokens=True, truncation=True)
                mask[-1].append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))

                tokens += [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
                inputx[-1].append(tokens)

            if temp["label"] == "B":
                labels.append(0)
            else:
                labels.append(1)

        global_att = np.zeros((len(data), 3, self.max_len), dtype=np.int32)
        global_att[:,:,0] = 1
        return {
            "inputx": torch.LongTensor(inputx),
            "mask": torch.LongTensor(mask),
            "label": torch.LongTensor(labels),
            "global_att": torch.LongTensor(global_att),
        }
