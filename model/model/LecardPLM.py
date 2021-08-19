import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from transformers import BertModel, AutoModel, AutoConfig
from model.DimReduction.DimRedBERT import DimRedBertModel
from tools.accuracy_tool import multi_label_accuracy, single_label_top1_accuracy


class LecardPLM(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(LecardPLM, self).__init__()

        plm_path = config.get('train', 'PLM_path')
        if "DimRedBERT" in plm_path:
            self.encoder = DimRedBertModel.from_pretrained(plm_path)
            self.plm_config = self.encoder.config
        else:
            self.encoder = AutoModel.from_pretrained(plm_path)
            self.plm_config = AutoConfig.from_pretrained(plm_path)
        self.lfm = 'Longformer' in self.plm_config.architectures[0]

        self.hidden_size = self.plm_config.hidden_size
        self.fc = nn.Linear(self.hidden_size, 2)

        self.criterion = nn.CrossEntropyLoss()

    def init_multi_gpu(self, device, config, *args, **params):
        self.encoder = nn.DataParallel(self.encoder, device_ids=device)
        self.fc = nn.DataParallel(self.fc, device_ids=device)
        print('init multi gpus')

    def forward(self, data, config, gpu_list, acc_result, mode):
        inputx = data['inputx']
        if self.lfm:
            # out = self.encoder(inputx, attention_mask = data['mask'], token_type_ids = data["segment"], global_attention_mask = data["global_att"])
            out = self.encoder(inputx, attention_mask = data['mask'], global_attention_mask = data["global_att"])
        else:
            # out = self.encoder(inputx, attention_mask = data['mask'], token_type_ids = data["segment"])
            out = self.encoder(inputx, attention_mask = data['mask'])
        y = out['pooler_output']
        result = self.fc(y)
        loss = self.criterion(result, data["labels"])
        acc_result = accuracy(result, data["labels"], acc_result)

        if mode == "train":
            return {"loss": loss, "acc_result": acc_result}
        else:
            score = torch.softmax(result, dim = 1) # batch, 2
            return {"loss": loss, "acc_result": acc_result, "score": score[:,1].tolist(), "index": data["index"]}


def accuracy(logit, label, acc_result):
    if acc_result is None:
        acc_result = {'right': 0, 'actual_num': 0, 'pre_num': 0}
    pred = torch.max(logit, dim = 1)[1]
    acc_result['pre_num'] += int((pred == 1).sum())
    acc_result['actual_num'] += int((label == 1).shape[0])
    acc_result['right'] += int((pred[label == 1] == 1).sum())
    return acc_result
