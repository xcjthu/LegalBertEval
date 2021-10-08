from .model.LecardPLM import LecardPLM
from .model.PairwiseLecardPLM import PairwisePLM
from .MemBERT.RecurrentTrans import RecurrentTransInf
model_list = {
    "lecard": LecardPLM,
    "pairwise": PairwisePLM,
    "recurrent": RecurrentTransInf,
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
