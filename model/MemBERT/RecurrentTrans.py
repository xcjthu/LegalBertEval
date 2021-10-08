from model.MemBERT.MemBERT import MemBertForMaskedLM, MemBertModel
from torch import nn
import torch
from transformers import AutoConfig

class RecurrentTransInf(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super().__init__()
        self.plm_config = AutoConfig.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.plm_config.mem_size = config.getint("train", "mem_size")

        self.encoder = MemBertModel.from_pretrained("hfl/chinese-roberta-wwm-ext", config=self.plm_config)

        self.hidden_size = self.plm_config.hidden_size
        self.fc = nn.Linear(self.hidden_size, 2)

        self.criterion = nn.CrossEntropyLoss()

    def init_multi_gpu(self, device, config, *args, **params):
        self.encoder = nn.DataParallel(self.encoder, device_ids=device)
        self.fc = nn.DataParallel(self.fc, device_ids=device)
        print('init multi gpus')
    
    def forward(self, data, config, gpu_list, acc_result, mode):
        mem = None
        inpb, inpb_mask = torch.transpose(data["inp"], 0, 1).contiguous(), torch.transpose(data["mask"], 0, 1).contiguous()
        block_num = inpb.shape[0]
        all_mem = []
        for i in range(block_num):
            out = self.encoder(inpb[i], attention_mask=inpb_mask[i], mem=mem, return_dict=False)
            mem = out[2] # batch, mem_size, hidden_size
            all_mem.append(mem)
        y = torch.max(torch.cat(all_mem, dim = 1), dim=1)[0]
        
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
