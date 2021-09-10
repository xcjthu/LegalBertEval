import json
import os
import re
from tqdm import tqdm

path = "/data2/private/xcj/PrivateLendingData/mjjd_data"

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
    ret = {}
    fin = open("/data/disk1/private/xcj/MJJDInfoExtract/LawPrediction/data/labels/usedlaws.json", "r")
    # fin = open("/data/disk1/private/xcj/MJJDInfoExtract/LawPrediction/data/lawmap/mapping.json", "r")
    alllaws = json.load(fin) + [law["target"] for law in json.load(open("/data/disk1/private/xcj/MJJDInfoExtract/LawPrediction/data/lawmap/mapping.json", "r"))]
    for law in alllaws:
        tk = "第%s条第%s款" % (num2cn(law["tiao"]), num2cn(law["kuan"])) if law["kuan"] != 0 else "第%s条" % num2cn(law["tiao"])
        tk = tk.replace("第一十", "第十")
        if law["law"] not in ret:
            ret[law["law"]] = set()
        ret[law["law"]].add(tk)
    for law in ret:
        ret[law] = list(ret[law])
    return ret
laws = read_labels()

print(json.dumps(laws, ensure_ascii=False))
data = []
total = 0
for i in range(20):
    print("thread_%s" % i)
    for fn in tqdm(os.listdir(os.path.join(path, "thread_%s" % i))):
        if not fn[-5:] == "jsonl":
            continue
        fin = open(os.path.join(path, "thread_%s" % i, fn), "r")
        for line in fin:
            case = json.loads(line)
            if (not case["attrs"]["judgement_date"] is None) and case["attrs"]["case_no"][1:5] != "2021":
                continue
            if (not case["attrs"]["judgement_type"] == "判决"):
                continue
            total += 1
            reflaw = []
            for l in case["law_cases"]:
                for gl in laws:
                    if gl in l["law_name_new"]:
                        if "款" in l["law_clause"]:
                            tk = l["law_clause"].split("款")[0] + "款"
                        else:
                            tk = l["law_clause"].split("条")[0] + "条"
                        if tk in laws[gl]:
                            reflaw.append({"law": gl, "tk": tk})
            if len(reflaw) > 0:
                case["reflaw"] = reflaw
                data.append(case)
    print(len(data), total)

fout = open("/data/disk1/private/xcj/MJJDInfoExtract/LawPrediction/data/cases.json", "w")
print(json.dumps(data, ensure_ascii=False, indent=2), file=fout)
fout.close()
