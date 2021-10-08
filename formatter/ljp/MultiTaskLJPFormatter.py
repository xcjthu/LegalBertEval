import json
import torch
import os
import numpy as np

from transformers import AutoTokenizer,BertTokenizer
from formatter.Basic import BasicFormatter


class MultiTaskLJPFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = AutoTokenizer.from_pretrained(config.get('train', 'PLM_vocab'))
        self.max_len = config.getint("train", "max_len")
        self.mode = mode

        self.ms = False
        try:
            self.ms = config.getboolean("data", "ms")
        except:
            pass

        label2id = json.load(open(config.get("data", "label2id"), "r"))
        self.charge2id = label2id["ac"] if self.ms else label2id["charge"]
        self.article2id = label2id["laws"]
    
    def process_ms(self, data, config, mode):
        inputx = []
        mask = []

        charge = []
        article = []

        for temp in data:
            tokens = self.tokenizer.encode(temp["fact"], max_length=self.max_len, add_special_tokens=True, truncation=True)
            mask.append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))
            tokens += [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
            inputx.append(tokens)
            if mode == "test":
                continue
            # temp_charge = np.zeros(len(self.charge2id), dtype=np.int)
            # for c in temp["charge"]:
            #     temp_charge[self.charge2id[str(c)]] = 1
            charge.append(self.charge2id[str(temp["charge"])])

            temp_article = np.zeros(len(self.article2id), dtype=np.int)
            for law in temp["laws"]:
                temp_article[self.article2id[law]] = 1
            article.append(temp_article.tolist())

        global_att = np.zeros((len(data), self.max_len), dtype=np.int32)
        global_att[:,0] = 1
        if mode == "test":
            return {
                "text": torch.LongTensor(inputx),
                "mask": torch.LongTensor(mask),
                "global_att": torch.LongTensor(global_att),
                "uids": [doc["uid"] for doc in data]
            }
        else:
            return {
                "text": torch.LongTensor(inputx),
                "mask": torch.LongTensor(mask),
                "charge": torch.LongTensor(charge),
                "law": torch.LongTensor(article),
                "global_att": torch.LongTensor(global_att),
            }

    def process(self, data, config, mode, *args, **params):
        if self.ms:
            return self.process_ms(data, config, mode)
        inputx = []
        mask = []

        charge = []
        article = []
        term = []

        for temp in data:
            tokens = self.tokenizer.encode(temp["fact"], max_length=self.max_len, add_special_tokens=True, truncation=True)
            mask.append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))
            tokens += [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
            inputx.append(tokens)

            if mode == "test":
                continue
            temp_charge = np.zeros(len(self.charge2id), dtype=np.int)
            for c in temp["charge"]:
                temp_charge[self.charge2id[str(c)]] = 1
            charge.append(temp_charge.tolist())

            temp_article = np.zeros(len(self.article2id), dtype=np.int)
            for law in temp["laws"]:
                temp_article[self.article2id[law]] = 1
            article.append(temp_article.tolist())

            if temp["imprisonment"]["life_imprisonment"]:
                temp_term = 350
            elif temp["imprisonment"]["death_penalty"]:
                temp_term = 400
            else:
                temp_term = int(temp["imprisonment"]["imprisonment"])

            term.append(temp_term)

        global_att = np.zeros((len(data), self.max_len), dtype=np.int32)
        global_att[:,0] = 1
        if mode == "test":
            return {
                "text": torch.LongTensor(inputx),
                "mask": torch.LongTensor(mask),
                "global_att": torch.LongTensor(global_att),
                "uids": [doc["uid"] for doc in data],
            }
        else:
            return {
                "text": torch.LongTensor(inputx),
                "mask": torch.LongTensor(mask),
                "charge": torch.LongTensor(charge),
                "law": torch.LongTensor(article),
                "term": torch.FloatTensor(term),
                "global_att": torch.LongTensor(global_att),
            }
