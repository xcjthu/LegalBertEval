import json
import random
import os
from tqdm import tqdm
import random

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


label2num = json.load(open("/home/ubuntu/mnt/LawPrediction/LawPrediction/data/labels/law2num.json", "r"))
label2id = {}
for key in label2num:
    if label2num[key] < 15 or key == "民法典-第六百七十五条":
        continue
    label2id[key] = len(label2id)

fout = open("/home/ubuntu/mnt/LawPrediction/LawPrediction/data/final_data/label2id.json", "w")
print(json.dumps(label2id, ensure_ascii=False, indent=2), file=fout)
fout.close()


path = "/home/ubuntu/mnt/LawPrediction/LawPrediction/data/case_mapper"
label2caseid = {key: [] for key in label2id}
for i in range(21):
    print(i)
    if i == 20:
        fin = open("/home/ubuntu/mnt/LawPrediction/LawPrediction/data/cases.json", "r")
    else:
        fin = open(os.path.join(path, "case%s.json" % i), "r")
    for cid, case in enumerate(json.load(fin)):
        for law in case["reflaw"]:
            tk = "第%s条第%s款" % (num2cn(law["tiao"]), num2cn(law["kuan"])) if law["kuan"] != 0 else "第%s条" % num2cn(law["tiao"])
            tk = tk.replace("第一十", "第十")
            key = "%s-%s" % (law["law"], tk)
            if key in label2caseid:
                label2caseid[key].append((i, cid))

caseids = set()
for key in label2caseid:
    if len(label2caseid[key]) < 1000:
        samples = label2caseid[key]
    else:
        samples = random.sample(label2caseid[key], 1000)
    caseids.update(samples)

print("the final number of cases", len(caseids))
cases = [[] for i in range(21)]
for c in caseids:
    cases[c[0]].append(c[1])
path = "/home/ubuntu/mnt/LawPrediction/LawPrediction/data/case_mapper"
gcases = []
for i in range(21):
    if i == 20:
        fin = open("/home/ubuntu/mnt/LawPrediction/LawPrediction/data/cases.json", "r")
    else:
        fin = open(os.path.join(path, "case%s.json" % i), "r")
    tmpcase = json.load(fin)
    for cid in cases[i]:
        gcases.append({"segments": tmpcase[cid]["segments"], "reflaw": tmpcase[cid]["reflaw"], "attrs": tmpcase[cid]["attrs"]})

fout = open("/home/ubuntu/mnt/LawPrediction/LawPrediction/data/final_data/final_cases.json", "w")
print(json.dumps(gcases, ensure_ascii=False, indent=2), file=fout)
fout.close()
