import json
import os
from torch.utils.data import Dataset
import numpy as np
import random

class xsMultiTaskDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        data_path = config.get('data', '%s_data' % mode)
        self.data = []
        for doc in json.load(open(data_path, 'r')):
            if doc["SS"] < 50:
                continue
            self.data.append({"fact": doc["SS"], "charge": doc["crime"], "imprisonment": doc["term_of_imprisonment"], "law": doc["related_laws"]})

        print("=" * 20, mode, "=" * 20)
        print("doc num", len(self.data))
        print('=' * 45)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
