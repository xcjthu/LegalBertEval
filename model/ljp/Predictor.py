import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class LJPPredictor(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(LJPPredictor, self).__init__()

        self.hidden_size = params["hidden_size"]

        self.ms = False
        try:
            self.ms = config.getboolean("data", "ms")
        except:
            pass

        label2id = json.load(open(config.get("data", "label2id"), "r"))
        if self.ms:
            self.charge_fc = nn.Linear(self.hidden_size, len(label2id["ac"]))
            self.article_fc = nn.Linear(self.hidden_size, len(label2id["laws"]) * 2)
        else:
            self.charge_fc = nn.Linear(self.hidden_size, len(label2id["charge"]) * 2)
            self.article_fc = nn.Linear(self.hidden_size, len(label2id["laws"]) * 2)
            self.term_fc = nn.Linear(self.hidden_size, 1)

    def init_multi_gpu(self, device, config, *args, **params):
        pass

    def forward(self, h):
        batch = h.size()[0]

        charge = self.charge_fc(h)
        article = self.article_fc(h)
        article = article.view(batch, -1, 2)
        if self.ms:
            return {"charge": charge, "law": article}
        else:
            charge = charge.view(batch, -1, 2)
            term = self.term_fc(h)
            term = term.view(batch)

            return {"charge": charge, "law": article, "term": term}
