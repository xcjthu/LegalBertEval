import json
from torch.utils.data import Dataset
import random
from pyltp import SentenceSplitter
import os

class zyjdTestDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        self.data_path = config.get("data", "%s_data_path" % mode)
        fnames = os.listdir(self.data_path)
        self.data = []
        for fn in fnames:
            fin = open(os.path.join(self.data_path, fn), "r")
            for line in fin:
                line = json.loads(line)
                if "本院认为" not in line["seg_content"]:
                    continue
                content = "".join(line["seg_content"]["本院认为"])
                caseid = str(line["case_id"])
                sents = SentenceSplitter.split(content)
                self.data.extend([{"sent": s, "id": caseid} for s in sents])

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
