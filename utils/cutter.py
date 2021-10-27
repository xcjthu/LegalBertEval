import json
import jieba
import os
from tqdm import tqdm

inpath = "/home/ubuntu/mnt/LawPrediction/LawPrediction/data/final_data"
word2num = {}
for fname in ["train.json", "test.json"]:
    fin = open(os.path.join(inpath, fname), "r")
    data = json.load(fin)
    fin.close()
    for did, doc in enumerate(tqdm(data)):
        text = []
        for sid, seg in enumerate(doc["segments"]):
            txt = [list(jieba.cut(t)) for t in seg["txt"]]
            data[did]["segments"][sid]["txt"] = txt
            for para in txt:
                for word in para:
                    if word not in word2num:
                        word2num[word] = 0
                    word2num[word] += 1
        # if did > 100:
        #     break
    fout = open(os.path.join(inpath, "cutted", fname), "w")
    print(json.dumps(data, ensure_ascii=False, indent=2), file=fout)
    fout.close()

fout = open(os.path.join(inpath, "cutted", "word2id.json"), "w")
word2id = {"[PAD]": 0, "[UNK]": 1}
for word in word2num:
    if word2num[word] > 20:
        word2id[word] = len(word2id)
print(json.dumps(word2id, ensure_ascii=False, indent=2), file=fout)
fout.close()
