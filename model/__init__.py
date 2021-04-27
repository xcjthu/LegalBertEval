from model.scm.Bert import SCMBert
from model.scm.ConcatBert import ConcatSCMBert

model_list = {
    "SCMBert": SCMBert,
    "ConcatSCMBert": ConcatSCMBert,
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
