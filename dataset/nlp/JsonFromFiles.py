import json
import os
from torch.utils.data import Dataset
import random
from tools.dataset_tool import dfs_search

class JsonFromFilesDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.file_list = []
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.encoding = encoding
        self.ms = False
        try:
            self.ms = config.getboolean("data", "ms")
        except:
            pass

        filename_list = config.get("data", "%s_file_list" % mode).replace(" ", "").split(",")
        recursive = False

        for name in filename_list:
            self.file_list = self.file_list + dfs_search(os.path.join(self.data_path, name), recursive)
        self.file_list.sort()

        self.data = []
        for filename in self.file_list:
            f = open(filename, "r", encoding=encoding)
            docs = json.load(f)
            for doc in docs:
                if mode == "test":
                    self.data.append({"fact": doc["SS"], "uid": doc["uid"]})
                    continue
                if not self.ms:
                    self.data.append({
                        "fact": doc["SS"],
                        "charge": doc["crime"],
                        "laws": ["%s_%s" % (law[0], law[1]) for law in doc["related_laws"]],
                        "imprisonment": doc["term_of_imprisonment"]
                    })
                else:
                    self.data.append({
                        "fact": doc["SS"],
                        "charge": doc["actioncause"],
                        "laws": ["%s_%s" % (law[0], law[1]) for law in doc["related_laws"]],
                    })

        if mode == "train":
            random.shuffle(self.data)

        self.reduce = config.getboolean("data", "reduce")
        if mode != "train":
            self.reduce = False
        if self.reduce:
            self.reduce_ratio = config.getfloat("data", "reduce_ratio")
        print("=" * 10, mode, "data num", len(self.data), '=' * 20)

    def __getitem__(self, item):
        if self.reduce:
            return self.data[random.randint(0, len(self.data) - 1)]
        return self.data[item]

    def __len__(self):
        if self.reduce:
            return int(self.reduce_ratio * len(self.data))
        return len(self.data)
