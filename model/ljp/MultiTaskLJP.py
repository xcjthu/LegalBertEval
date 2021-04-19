import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from model.loss import MultiLabelSoftmaxLoss, log_square_loss
from model.ljp.Predictor import LJPPredictor
from tools.accuracy_tool import multi_label_accuracy, log_distance_accuracy_function
from transformers import BertModel, AutoModel, AutoConfig

class MultiTaskLJP(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(MultiTaskLJP, self).__init__()
        plm_path = config.get('train', 'PLM_path')
        
        self.encoder = AutoModel.from_pretrained(plm_path)
        self.plm_config = AutoConfig.from_pretrained(plm_path)
        self.lfm = 'Longformer' in self.plm_config.architectures[0]
        
        self.hidden_size = self.plm_config.hidden_size

        self.fc = LJPPredictor(config, gpu_list, *args, hidden_size=self.hidden_size)

        label2id = json.load(open(config.get("data", "label2id"), "r"))
        self.criterion = {
            "charge": MultiLabelSoftmaxLoss(config, len(label2id["charge"])),
            "law": MultiLabelSoftmaxLoss(config, len(label2id["law"])),
            "term": log_square_loss
        }
        self.accuracy_function = {
            "charge": multi_label_accuracy,
            "law": multi_label_accuracy,
            "term": log_distance_accuracy_function,
        }

    def init_multi_gpu(self, device, config, *args, **params):
        self.encoder = nn.DataParallel(self.encoder, device_ids=device)
        self.fc = nn.DataParallel(self.fc, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['text']
        if self.lfm:
            out = self.encoder(x, attention_mask = data['mask'], global_attention_mask = data["global_att"])
        else:
            out = self.encoder(x, attention_mask = data['mask'])
        y = out['pooler_output']
        result = self.fc(y)

        loss = 0
        for name in ["charge", "law", "term"]:
            loss += self.criterion[name](result[name], data[name])

        if acc_result is None:
            acc_result = {"charge": None, "law": None, "term": None}

        for name in ["charge", "law", "term"]:
            acc_result[name] = self.accuracy_function[name](result[name], data[name], config, acc_result[name])

        return {"loss": loss, "acc_result": acc_result}
