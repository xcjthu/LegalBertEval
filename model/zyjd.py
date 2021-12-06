import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from model.loss import MultiLabelSoftmaxLoss, log_sum_exp

from tools.accuracy_tool import prf
from transformers import BertModel, AutoModel, AutoConfig
from torch.autograd import Variable


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, :11] # batch, 11, hidden_size
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output.view(hidden_states.shape[0], -1)

class zyjd(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(zyjd, self).__init__()
        plm_path = config.get('train', 'PLM_path')

        self.encoder = AutoModel.from_pretrained(plm_path)
        self.plm_config = AutoConfig.from_pretrained(plm_path)
        self.hidden_size = self.plm_config.hidden_size
        self.pooler = BertPooler(self.plm_config)

        self.read_label(config)
        self.fc = nn.Linear(self.hidden_size * 11, len(self.label2id))
        # self.criterion = MultiLabelSoftmaxLoss(config, len(label2id))
        # self.criterion = nn.CrossEntropyLoss()
        # self.accuracy_function = prf
        self.accuracy_function = multilabel_prf
        self.criterion = log_sum_exp
        # self.criterion = nn.BCEWithLogitsLoss()

    def read_label(self, config):
        label2num = json.load(open(config.get("data", "label2num")))
        num_threshold = config.getint("data", "threshold")
        self.label2id = {}#{"NA": 0}
        deletelabel = set([line.strip() for line in open("/data/disk1/private/xcj/MJJDInfoExtract/DisputeFocus/data/zyjd/delete_label.txt", "r")])
        for l in label2num:
            if l in deletelabel:
                continue
            self.label2id[l] = len(self.label2id)
            # if label2num[l] > num_threshold:
            #     # key = "/".join(l.split("/")[:2])
            #     # key = l.split("/")[0]
            #     key = l
            #     if key not in self.label2id:
            #         self.label2id[key] = len(self.label2id)

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['text']
        batch = x.shape[0]
        # if self.lfm:
        #     out = self.encoder(x, attention_mask = data['mask'], global_attention_mask = data["global_att"])
        # else:
        out = self.encoder(x, attention_mask = data['mask'])
        # y = out['pooler_output']
        y = self.pooler(out["last_hidden_state"])
        # result = self.fc(y).view(batch, -1, 2)
        result = self.fc(y).view(batch, -1)
        # if mode == "train":
        #     result = result - 100 * data["label_mask"]
        loss = self.criterion(result, data["label"])
        acc_result = self.accuracy_function(result, data["label"], config, acc_result)

        return {"loss": loss, "acc_result": acc_result, "output": list(zip(torch.max(result, dim=1)[1].tolist(), data["ids"]))}

def multilabel_prf(output, label, config, acc_result):
    # result: batch, label_num
    # label: batch, label_num
    if acc_result is None:
        acc_result = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    pred = (output > 0).int()
    # id2 = label
    acc_result["TP"] += int((pred[label != 0] == label[label != 0]).sum())
    acc_result["TN"] += int((pred[label == 0] == 0).sum())
    acc_result["FP"] += int((pred[pred != 0] != label[pred != 0]).sum())
    acc_result["FN"] += int((label[pred == 0] != 0).sum())
    return acc_result
