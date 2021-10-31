import json
from torch.utils.data import Dataset

class LawDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        self.data_path = config.get("data", "%s_data_path" % mode)
        data = json.load(open(self.data_path, "r"))
        alll2id = json.load(open(config.get("data", "label2id"), "r"))
        # self.label2id = json.load(open(config.get("data", "label2id"), "r"))
        self.label2id = {}
        for key in alll2id:
            if alll2id[key] <=  100:
                self.label2id[key] = len(self.label2id)
        # self.label2id = json.load(open(config.get("data", "label2id"), "r"))
        self.data = []
        average_label_num = 0
        for doc in data:
            text = ""
            for seg in doc["segments"]:
                if seg["label"] in ["原告诉称", "被告辩称"]:
                    text += "\n".join(seg["txt"])
            if text == "":
                continue
            label = [law["law"] + "-" + ("第%s条第%s款" % (self.num2cn(law["tiao"]), self.num2cn(law["kuan"])) if law["kuan"] != 0 else "第%s条" % self.num2cn(law["tiao"])).replace("第一十", "第十") for law in doc["reflaw"]]
            label = [l for l in label if l in self.label2id]
            if len(label) == 0:
                continue
            average_label_num += len(label)
            self.data.append({"inp": text, "label": label})
        print(len(self.data), average_label_num / len(data))

    def num2cn(self, num):
        dic_shu = "零一二三四五六七八九十"
        s = []
        while len(s) < 4:
            s.append(num % 10)
            num = int(num/10)
        s.append(num)
        # print(s)

        hz = ""
        dw = ["", "十", "百", "千", "万"]
        i = 0
        while i < len(s):
            if s[i] != 0:
                if s[i] >= 10:
                    hz = self.num2cn(s[i]) + dw[i] + hz
                else:
                    hz = dic_shu[s[i]] + dw[i] + hz
            i += 1
        return hz

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
