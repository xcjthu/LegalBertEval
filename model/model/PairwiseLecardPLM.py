import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from transformers import BertModel, AutoModel, AutoConfig

from tools.accuracy_tool import multi_label_accuracy, single_label_top1_accuracy


class PairwisePLM(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(PairwisePLM, self).__init__()

        plm_path = config.get('train', 'PLM_path')
        
        self.encoder = AutoModel.from_pretrained(plm_path)
        self.plm_config = AutoConfig.from_pretrained(plm_path)
        self.lfm = 'Longformer' in self.plm_config.architectures[0]

        self.hidden_size = self.plm_config.hidden_size
        self.fc = nn.Linear(self.hidden_size, 1)

        self.criterion = nn.CrossEntropyLoss()

    def init_multi_gpu(self, device, config, *args, **params):
        self.encoder = nn.DataParallel(self.encoder, device_ids=device)
        self.fc = nn.DataParallel(self.fc, device_ids=device)
        print('init multi gpus')

    def forward(self, data, config, gpu_list, acc_result, mode):
        # inputx: batch, 2, seq_len
        pair = 2 if mode == "train" else 1
        batch, seq_len = data["inputx"].shape[0], data["inputx"].shape[2]
        inputx = data["inputx"].view(batch * pair, seq_len)
        mask = data["mask"].view(batch * pair, seq_len)
        segment = data["segment"].view(batch * pair, seq_len)
        global_att = data["global_att"].view(batch * pair, seq_len)
        if self.lfm:
            # out = self.encoder(inputx, attention_mask = mask, token_type_ids = segment, global_attention_mask = global_att)
            out = self.encoder(inputx, attention_mask = mask, global_attention_mask = global_att)
        else:
            # out = self.encoder(inputx, attention_mask = mask, token_type_ids = segment)
            out = self.encoder(inputx, attention_mask = mask)
        y = out['pooler_output'].squeeze(1)
        result = self.fc(y).view(batch, pair)

        if mode == "train":
            loss = self.criterion(result, data["labels"])
            acc_result = accuracy(result, data["labels"], acc_result)
            return {"loss": loss, "acc_result": acc_result}
        else:
            acc_result = {"right": 0, "total": 0}
            return {"loss": 0, "acc_result": acc_result, "score": y.tolist(), "index": data["index"]}


def accuracy(logit, label, acc_result):
    if acc_result is None:
        acc_result = {'right': 0, 'total': 0}
    pred = torch.max(logit, dim = 1)[1]
    acc_result['total'] += int(label.shape[0])
    acc_result['right'] += int((pred == label).sum())
    return acc_result
