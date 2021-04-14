import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,BertForMultipleChoice,LongformerForMultipleChoice

from tools.accuracy_tool import single_label_top1_accuracy


class MyBertQA(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(MyBertQA, self).__init__()
        bert_path = config.get("model", "bert_path")
        if bert_path not in ['bert-base-chinese', 'hfl/chinese-roberta-wwm-ext', '/mnt/datadisk0/xcj/LegalBert/model/LRoBERTa']:
            self.bert = LongformerForMultipleChoice.from_pretrained(config.get("model", "bert_path"))
            self.lfm = True
        else:
            self.bert = BertForMultipleChoice.from_pretrained(config.get("model", "bert_path"))
            self.lfm = False

        self.multi = config.getboolean("data", "multi_choice")
        if self.multi:
            self.criterion = nn.BCEWithLogitsLoss()
            self.accuracy_function = multi_accuracy
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.accuracy_function = single_accracy

    def init_multi_gpu(self, device, config, *args, **params):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        if self.lfm:
            ret = self.bert(data["input"], attention_mask=data["mask"], token_type_ids=data["segment"], global_attention_mask=data["global_att"])
        else:
            ret = self.bert(data["input"], attention_mask=data["mask"], token_type_ids=data["segment"])
        logit = ret["logits"]
        if self.multi:
            loss = self.criterion(logit, data["label"].float())
        else:
            loss = self.criterion(logit, data["label"])

        label = data["label"]
        acc_result = self.accuracy_function(logit, label, acc_result)
        return {"loss": loss, "acc_result": acc_result}

def single_accracy(logit, label, acc_result):
    if acc_result is None:
        acc_result = {'right': 0, 'total': 0}
    pred = torch.max(logit, dim = 1)[1]
    acc_result['total'] += int(logit.shape[0])
    acc_result['right'] += int((pred == label).sum())
    return acc_result

def multi_accuracy(logit, label, acc_result):
    # logit: batch, option_num (Float)
    # label: batch, option_num (Long)
    if acc_result is None:
        acc_result = {'right': 0, 'total': 0}
    pred = (logit > 0).long()
    acc_result['total'] += int(logit.shape[0])
    for i in range(pred.shape[0]):
        if (pred[i] == label[i]).all():
            acc_result['right'] += 1
    return acc_result

