import json
import random
import os

qpath = "/home/xcj/LegalLongPLM/data/LeCaRD-main/data/query/query.json"
outpath = "/home/xcj/LegalLongPLM/data/LeCaRD-main/data/query_split"

data = [json.loads(line) for line in open(qpath, "r")]
random.shuffle(data)

for i in range(5):
    datai = data[i::5]
    fout = open(os.path.join(outpath, "query_%d.json" % i), "w")
    print(json.dumps(datai, ensure_ascii=False, indent=2), file=fout)
    fout.close()
