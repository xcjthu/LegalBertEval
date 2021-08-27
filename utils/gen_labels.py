import json
import xlrd

path = "/data/disk1/private/xcj/MJJDInfoExtract/LawPrediction/data/labels/自动推理表格汇总.xlsx"

data = xlrd.open_workbook(path)
table = data.sheets()[0]
nrows = table.nrows

laws = []
l2 = None
for row in range(2, nrows):
    title = table.cell_value(row, 1)
    tk_ = str(table.cell_value(row, 2)).replace(",", "、").replace("，", "、")
    if tk_ == "":
        continue
    if title != "":
        if title[0] == "《":
            title = title[1:]
        if title[-1] == "》":
            title = title[:-1]
        l2 = title
    for tk in tk_.split("、"):
        tk = tk.split(".")
        tiao = tk[0]
        kuan = tk[1] if len(tk) == 2 else 0
    laws.append({"law": l2, "tiao": int(tiao), "kuan": int(kuan)})

print(len(laws))
laws.sort(key=lambda x:x["law"])
fout = open("/data/disk1/private/xcj/MJJDInfoExtract/LawPrediction/data/labels/usedlaws.json", "w")
print(json.dumps(laws, ensure_ascii=False, indent=2), file=fout)
fout.close()
