import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
from tqdm import tqdm

texts = []
path = "/home/xcj/LegalLongPLM/data/LeCaRD-main/data/candidates/candidates"
for qid in os.listdir(path):
    for fn in os.listdir(os.path.join(path, qid)):
        case = json.load(open(os.path.join(path, qid, fn), "r"))
        texts.append(case["ajjbqk"])

qpath = "/home/xcj/LegalLongPLM/data/LeCaRD-main/data/query"
for line in open(os.path.join(qpath, "query.json"), "r"):
    texts.append(json.loads(line)["q"])

import jieba
corpus = [" ".join(jieba.cut(text)) for text in tqdm(texts)]

vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
dump(vectorizer, '/home/xcj/LegalLongPLM/lecard/tfidf.joblib') 

