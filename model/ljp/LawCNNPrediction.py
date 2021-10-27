import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from model.loss import MultiLabelSoftmaxLoss, log_square_loss

from tools.accuracy_tool import multi_label_accuracy, log_distance_accuracy_function, single_label_top1_accuracy
from transformers import BertModel, AutoModel, AutoConfig

class CNNEncoder(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(CNNEncoder, self).__init__()

        self.emb_dim = config.getint("model", "hidden_Size")
        self.output_dim = self.emb_dim // 4

        self.min_gram = 2
        self.max_gram = 5
        self.convs = []
        for a in range(self.min_gram, self.max_gram + 1):
            self.convs.append(nn.Conv2d(1, self.output_dim, (a, self.emb_dim)))

        self.convs = nn.ModuleList(self.convs)
        self.feature_len = self.emb_dim
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size()[0]

        x = x.view(batch_size, 1, -1, self.emb_dim)

        conv_out = []
        gram = self.min_gram
        for conv in self.convs:
            y = self.relu(conv(x))
            y = torch.max(y, dim=2)[0].view(batch_size, -1)

            conv_out.append(y)
            gram += 1

        conv_out = torch.cat(conv_out, dim=1)

        return conv_out

class LawCNNPrediction(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(LawCNNPrediction, self).__init__()
        self.hidden_size = config.getint("model", "hidden_size")

        self.encoder = CNNEncoder(config, gpu_list, *args, **params)
        self.embedding = nn.Embedding(len(json.load(open(config.get("data", "word2id")))), self.hidden_size)

        label2id = json.load(open(config.get("data", "label2id"), "r"))
        self.fc = nn.Linear(self.hidden_size, len(label2id) * 2)

        self.accuracy_function = multi_label_accuracy

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['text']
        batch = x.shape[0]
        if self.lfm:
            out = self.encoder(x, attention_mask = data['mask'], global_attention_mask = data["global_att"])
        else:
            out = self.encoder(x, attention_mask = data['mask'])
        y = out['pooler_output']
        # result = self.fc(y).view(batch, -1, 2)
        result = self.fc(y).view(batch, -1)
        if mode == "train":
            result = result - 100 * data["label_mask"]
            loss = self.criterion(result, data["label"])
            acc_result = self.accuracy_function(reshape(result), data["alllabel"], config, acc_result)
        else:
            loss = 0
            acc_result = self.accuracy_function(reshape(result), data["alllabel"], config, acc_result)

        return {"loss": loss, "acc_result": acc_result}

def reshape(result):
    ret = torch.zeros(result.shape).to(result.device)
    ret[torch.arange(ret.shape[0]).unsqueeze(1).repeat(1, 2).to(result.device), torch.topk(result, 2)[1]] = 1
    return ret

