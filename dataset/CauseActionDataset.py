import json
import os
from torch.utils.data import Dataset
import numpy as np
import random

class CauseActionDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.max_len = config.getint("train", "max_len")

        data_path = config.get('data', '%s_data' % mode)
        allnames = os.listdir(data_path)
        allnames.sort()
        if mode == 'train':
            fnames = allnames[:int(0.7 * len(allnames))]
        else:
            fnames = allnames[int(0.7 * len(allnames)):]
        self.data = []
        self.label2id = json.load(open(config.get('data', 'label2id')))
        for fn in fnames:
            data = json.load(open(os.path.join(data_path, fn), 'r'))
            for doc in data:
                if len(doc['SS']) < 10:
                    continue
                lset = [str(l) for l in doc['AJAY'] if str(l) in self.label2id]
                if len(lset) == 0:
                    continue
                self.data.append({'text': doc['SS'], 'label': lset})

        doc_len = np.array([len(doc['text']) for doc in self.data])
        print('==' * 10, mode, 'dataset', '==' * 10)
        print('label num:', len(self.label2id))
        print('doc num:', len(self.data))
        print('average doc len', doc_len.mean(), 'max doc len', doc_len.max())
        print('%s docs are longer than the max len' % ((doc_len > self.max_len).sum() / len(self.data)))
        print('==' * 25)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
