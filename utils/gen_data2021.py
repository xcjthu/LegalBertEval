import json
import re
import os
from tqdm import tqdm

class LawMapper:
    def __init__(self):
        mapper = json.load(open("/data/disk1/private/xcj/MJJDInfoExtract/LawPrediction/data/lawmap/mapping.json", "r"))
        # self.mapper = {json.dumps(pair["origin"], ensure_ascii=False): json.dumps(pair["target"], ensure_ascii=False) for pair in mapper}
        self.common_used_numerals_tmp = {"〇": 0, '零':0, '一':1, '二':2, '两':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9, '十':10, '百':100, '千':1000, '万':10000, '亿':100000000}
        self.cared_law = {json.dumps(law, ensure_ascii = False) for law in json.load(open("/data/disk1/private/xcj/MJJDInfoExtract/LawPrediction/data/labels/usedlaws.json", "r"))}
        self.tk = re.compile(r"第?[零一二三四五六七八九十百0-9]+条(?:第?（?[零一二三四五六七八九十百0-9]）?[项款]){0,2}、?")

    def map_law(self, lawin):
        key = json.dumps(lawin, ensure_ascii=False)
        # if key in self.mapper:
        #     return json.loads(self.mapper[key])
        if key in self.cared_law:
            return lawin
        else:
            return None

    def chstring2int(self, uchar):
        ##1）按亿、万分割字符
        if uchar[0] in "0123456789":
            return int(uchar)
        if uchar[0] == "十":
            uchar = "一" + uchar
        sep_char = re.split(r'亿|万', uchar)
        total_sum = 0
        for i,sc in enumerate(sep_char):
            ##print("level 1:{}-----{}".format(i,sc))
            ##2）对每一部分进行转化处理，比如第二部分[ "三千二百四十二"]
            split_num = sc.replace('千', '1000').replace('百', '100').replace('十', '10')
            int_series = re.split(r'(\d{1,})', split_num)
            int_series.append("")
            int_series = ["".join(i) for i in zip(int_series[0::2],int_series[1::2])]
            int_series = ['零' if i == '' else i for i in int_series]
            num = 0
            ##int_series：["三1000", "二100", "四10", "二"]
            ##3）求和加总int_series
            for ix, it in enumerate(int_series):
                it = re.sub('零', '', it) if it != '零' else it
                ##print("level 2:{}{}".format(ix,it))

                temp = self.common_used_numerals_tmp[it[0]]*int(it[1:]) if len(it)>1 else self.common_used_numerals_tmp[it[0]]
                num += temp
                ##print("transformed part sum %s"%str(num))
            total_sum += num * (10 ** (4*(len(sep_char) - i - 1)))
        return total_sum

    def formatLname(self, lname):
        lname = re.sub(r"\s", "", lname).replace("《", "〈").replace("》", "〉").replace("中华人民共和国", "")
        lname = re.sub(r"[\(（].*?[）\)]", "", lname)
        return lname

    def formatTK(self, lname, tk):
        lname = self.formatLname(lname)

        tk = tk.replace("第", "").replace("(", "").replace(")", "").replace("（", "").replace("）", "").replace("〇", "")
        tksplit = tk.split("条")
        tiao = tksplit[0]
        if len(tksplit) > 1 and "款" in tksplit[1]:
            tksplit = tksplit[1].split("款")
            kuan = tksplit[0]
        else:
            kuan = "零"
        try:
            tiao = self.chstring2int(tiao)
            kuan = self.chstring2int(kuan)
        except:
            pass
        return {"law": lname, "tiao": tiao, "kuan": kuan}

    def process_laws(self, law_cases):
        ret = []
        for l in law_cases:
            ltk = self.formatTK(l["law_name_new"], l["law_clause"])
            ltkmap = self.map_law(ltk)
            if not ltkmap is None:
                ret.append(ltkmap)
        return ret

if __name__ == "__main__":
    path = "/data2/private/xcj/PrivateLendingData/mjjd_data"
    total = 0
    lawMapper = LawMapper()
    data = []
    for i in range(20):
        print("thread_%s" % i)
        for fn in tqdm(os.listdir(os.path.join(path, "thread_%s" % i))):
            if not fn[-5:] == "jsonl":
                continue
            fin = open(os.path.join(path, "thread_%s" % i, fn), "r")
            for line in fin:
                case = json.loads(line)
                if (not case["attrs"]["judgement_date"] is None) and case["attrs"]["judgement_date"][:4] != "2021":
                    continue
                if (not case["attrs"]["judgement_type"] == "判决"):
                    continue
                total += 1
                reflaw = lawMapper.process_laws(case["law_cases"])
                if len(reflaw) > 0:
                    case["reflaw"] = reflaw
                    data.append(case)
        print(len(data), total)

fout = open("/data/disk1/private/xcj/MJJDInfoExtract/LawPrediction/data/cases.json", "w")
print(json.dumps(data, ensure_ascii=False, indent=2), file=fout)
fout.close()
