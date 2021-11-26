import json
from torch.utils.data import Dataset
import random

class zyjdDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        self.data_path = config.get("data", "%s_data_path" % mode)

        data = json.load(open(self.data_path, "r"))
        self.positive = data["pos"]
        for i in range(len(self.positive)):
            self.positive[i]["id"] = "p-%d" % i
        self.negative = [{"sent": s, "label": ["NA"]} for s in data["neg"]]
        for i in range(len(self.negative)):
            self.negative[i]["id"] = "n-%d" % i
        self.read_label(config)
        if mode == "train":
            self.ratio = 0 #config.getfloat("train", "neg_ratio")
        else:
            self.ratio = 0
            self.data = self.positive # + self.negative

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

    def __getitem__(self, item):
        if self.mode == "train":
            if random.random() < len(self.positive) / len(self):
                return random.choice(self.positive)
            else:
                return random.choice(self.negative)
        else:
            return self.data[item]

    def __len__(self):
        return len(self.positive) + int(self.ratio * len(self.negative))
