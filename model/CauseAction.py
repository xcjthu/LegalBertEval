import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from transformers import BertModel,RobertaModel,AutoModel,AutoConfig
from tools.accuracy_tool import single_label_top1_accuracy

class CauseAction(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(CauseAction, self).__init__()

        self.encoder = AutoModel.from_pretrained(config.get('train', 'PLM_path'))
        self.plm_config = AutoConfig.from_pretrained(config.get('train', 'PLM_path'))

        label2id = json.load(open(config.get('data', 'label2id')))
        self.class_num = len(label2id)

        self.hidden_size = 768
        self.criterion = nn.CrossEntropyLoss()
        self.fc = nn.Linear(self.hidden_size, self.class_num)

        self.accuracy_function = single_label_top1_accuracy

    def forward(self, data, config, gpu_list, acc_result, mode):
        inputx = data['input']
        if self.plm_config.model_type == 'longformer':
            output = self.encoder(inputx, attention_mask=data['mask'], global_attention_mask=data['global_mask'])
        else:
            output = self.encoder(inputx, attention_mask=data['mask'])
        score = self.fc(output['pooler_output']) # batch, class_num
        loss = self.criterion(score, data["label"])
        #acc_result = acc(score, data['label'], acc_result)
        acc_result = self.accuracy_function(score, data["label"], config, acc_result)

        return {'loss': loss, 'acc_result': acc_result}

def acc(score, label, acc_result):
    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}
    predict = torch.max(score, dim = 1)[1]
    acc_result['total'] += int(label.shape[0])
    acc_result['right'] += int((predict == label).int().sum())
    return acc_result
