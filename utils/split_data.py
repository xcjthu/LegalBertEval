import json
import random
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np

def num2cn(num):
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
                hz = num2cn(s[i]) + dw[i] + hz
            else:
                hz = dic_shu[s[i]] + dw[i] + hz
        i += 1
    return hz

cases = json.load(open("/home/ubuntu/mnt/LawPrediction/LawPrediction/data/final_data/final_cases.json", "r"))
label2id = json.load(open("/home/ubuntu/mnt/LawPrediction/LawPrediction/data/final_data/label2id.json", "r"))

Xtext = [c["segments"] for c in cases]
X = np.arange(len(Xtext))
X.resize(len(Xtext), 1)
Y = np.zeros((len(Xtext), len(label2id)))
for cid, c in enumerate(cases):
    tmp = []
    for law in c["reflaw"]:
        tk = "第%s条第%s款" % (num2cn(law["tiao"]), num2cn(law["kuan"])) if law["kuan"] != 0 else "第%s条" % num2cn(law["tiao"])
        tk = tk.replace("第一十", "第十")
        key = "%s-%s" % (law["law"], tk)
        if key in label2id:
            Y[cid, label2id[key]] = 1

X_train, y_train, X_test, y_test = iterative_train_test_split(X, Y, test_size = 0.15)
case_train = [cases[x[0]] for x in X_train]
case_test = [cases[x[0]] for x in X_test]
fout = open("/home/ubuntu/mnt/LawPrediction/LawPrediction/data/final_data/train.json", "w")
print(json.dumps(case_train, ensure_ascii=False, indent=2), file=fout)
fout.close()

fout = open("/home/ubuntu/mnt/LawPrediction/LawPrediction/data/final_data/test.json", "w")
print(json.dumps(case_test, ensure_ascii=False, indent=2), file=fout)
fout.close()
