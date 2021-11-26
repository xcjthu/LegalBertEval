from .ljp.LawPrediction import LawPrediction
from .ljp.LawCNNPrediction import LawCNNPrediction
from .zyjd import zyjd
model_list = {
    "law": LawPrediction,
    "cnn": LawCNNPrediction,
    "zyjd": zyjd,
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
