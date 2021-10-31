import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from model.loss import MultiLabelSoftmaxLoss, log_square_loss

from tools.accuracy_tool import multi_label_accuracy, log_distance_accuracy_function, single_label_top1_accuracy
from transformers import BertModel, AutoModel, AutoConfig

class LawPrediction(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(LawPrediction, self).__init__()
        plm_path = config.get('train', 'PLM_path')

        self.encoder = AutoModel.from_pretrained(plm_path)
        self.plm_config = AutoConfig.from_pretrained(plm_path)
        self.lfm = 'Longformer' in self.plm_config.architectures[0]

        self.hidden_size = self.plm_config.hidden_size

        # label2id = json.load(open(config.get("data", "label2id"), "r"))
        alll2id = json.load(open(config.get("data", "label2id"), "r"))
        # label2id = json.load(open(config.get("data", "label2id"), "r"))
        label2id = {}
        for key in alll2id:
            if alll2id[key] <=  100:
                label2id[key] = len(label2id)
        self.fc = nn.Linear(self.hidden_size, len(label2id) * 2)
        # self.fc = nn.Linear(self.hidden_size, len(label2id))

        self.criterion = MultiLabelSoftmaxLoss(config, len(label2id))
        # self.criterion = nn.CrossEntropyLoss()
        self.accuracy_function = multi_label_accuracy

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['text']
        batch = x.shape[0]
        if self.lfm:
            out = self.encoder(x, attention_mask = data['mask'], global_attention_mask = data["global_att"])
        else:
            out = self.encoder(x, attention_mask = data['mask'])
        y = out['pooler_output']
        result = self.fc(y).view(batch, -1, 2)
        # result = self.fc(y).view(batch, -1)
        # if mode == "train":
        #     result = result - 100 * data["label_mask"]
        loss = self.criterion(result, data["label"])
        acc_result = self.accuracy_function(result, data["label"], config, acc_result)
        # else:
        #     loss = 0
        #     acc_result = self.accuracy_function(reshape(result), data["alllabel"], config, acc_result)

        return {"loss": loss, "acc_result": acc_result}

def reshape(result):
    ret = torch.zeros(result.shape).to(result.device)
    ret[torch.arange(ret.shape[0]).unsqueeze(1).repeat(1, 2).to(result.device), torch.topk(result, 2)[1]] = 1
    return ret

