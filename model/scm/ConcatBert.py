import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.accuracy_tool import single_label_top1_accuracy
from transformers import AutoModel, AutoConfig

class ConcatSCMBert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(ConcatSCMBert, self).__init__()
        plm_path = config.get('train', 'PLM_path')

        self.encoder = AutoModel.from_pretrained(plm_path)
        self.plm_config = AutoConfig.from_pretrained(plm_path)
        self.lfm = 'Longformer' in self.plm_config.architectures[0]

        self.hidden_size = self.plm_config.hidden_size
        
        self.fc = nn.Linear(self.hidden_size, 1)

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_function = single_label_top1_accuracy

    def init_multi_gpu(self, device, config, *args, **params):
        self.encoder = nn.DataParallel(self.encoder, device_ids=device)
        self.fc = nn.DataParallel(self.fc, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        batch = data["inputx"].shape[0]
        intputx = data["inputx"].view(batch * 2, -1)
        mask = data["mask"].view(batch * 2, -1)
        gat = data["global_att"].view(batch * 2, -1)

        if self.lfm:
            out = self.encoder(intputx, attention_mask = mask, global_attention_mask = gat)
        else:
            out = self.encoder(intputx, attention_mask = mask)
        y = out['pooler_output'].view(batch, 2, self.hidden_size)
        s = self.fc(y).squeeze(2)

        loss = self.criterion(s, data["label"])
        acc_result = self.accuracy_function(s,  data["label"], config, acc_result)

        return {"loss": loss, "acc_result": acc_result}
