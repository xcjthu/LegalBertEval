import json
import os
from tqdm import tqdm

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

def read_labels():
    law2num = {}
    fin = open("/data/disk1/private/xcj/MJJDInfoExtract/LawPrediction/data/labels/usedlaws.json", "r")
    for law in json.load(fin):
        tk = "第%s条第%s款" % (num2cn(law["tiao"]), num2cn(law["kuan"])) if law["kuan"] != 0 else "第%s条" % num2cn(law["tiao"])
        tk = tk.replace("第一十", "第十")
        law2num["%s-%s" % (law["law"], tk)] = 0
    return law2num
law2num = read_labels()


path = "/data/disk1/private/xcj/MJJDInfoExtract/LawPrediction/data/case_mapper"
for i in range(21):
    if i == 20:
        fin = open("/data/disk1/private/xcj/MJJDInfoExtract/LawPrediction/data/cases.json", "r")
    else:
        fin = open(os.path.join(path, "case%s.json" % i), "r")
    for case in tqdm(json.load(fin)):
        for law in case["reflaw"]:
            tk = "第%s条第%s款" % (num2cn(law["tiao"]), num2cn(law["kuan"])) if law["kuan"] != 0 else "第%s条" % num2cn(law["tiao"])
            tk = tk.replace("第一十", "第十")
            key = "%s-%s" % (law["law"], tk)
            if key not in law2num:
                law2num[key] = 0
            law2num[key] += 1


fout = open("/data/disk1/private/xcj/MJJDInfoExtract/LawPrediction/data/labels/law2num.json", "w")
print(json.dumps(dict(sorted(law2num.items(), key=lambda x:x[1], reverse=True)), ensure_ascii=False, indent=2), file=fout)
fout.close()
