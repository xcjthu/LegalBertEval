import json

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

dpath = "/data/disk1/private/xcj/MJJDInfoExtract/LawPrediction/data/cases.json"
fin = open(dpath, "r")
for case in json.load(fin):
    for l in case["reflaw"]:
        law2num["%s-%s" % (l["law"], l["tk"])] += 1

fout = open("law2num.json", "w")
print(json.dumps(dict(sorted(law2num.items(), key=lambda x:x[1], reverse=True)), ensure_ascii=False, indent=2), file=fout)
fout.close()
