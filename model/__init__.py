from .model.LecardPLM import LecardPLM
from .model.PairwiseLecardPLM import PairwisePLM

model_list = {
    "lecard": LecardPLM,
    "pairwise": PairwisePLM,
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
