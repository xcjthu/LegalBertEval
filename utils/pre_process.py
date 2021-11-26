import json

data = json.load(open("../../data/民间借贷要素标注_1031.json", "r"))["term"]

docs = []
labelset = set()
for doc in data:
    content = doc["content"]["content"]
    labels = [{"content": l["content"], "label": "/".join(l["value"]), "pos": [l["start"], l["end"]]} for l in doc["result"]["answer"]]
    docs.append({"content": content, "label": labels})
    for l in labels:
        labelset.add(l["label"])
label2id = {l: i for i, l in enumerate(labelset)}
fout = open("../../data/label2id.json", "w")
print(json.dumps(label2id, ensure_ascii=False, indent=2), file=fout)
fout.close()

fout = open("../../data/data.json", "w")
print(json.dumps(docs, ensure_ascii=False, indent=2), file=fout)
fout.close()
