from model.qa.Bert import BertQA
from model.qa.BiDAF import BiDAFQA
from model.qa.CoMatch import CoMatching
from model.qa.HAF import HAF
from model.qa.MyBertQA import MyBertQA

model_list = {
    "BertQA": BertQA,
    "BiDAFQA": BiDAFQA,
    "Comatch": CoMatching,
    "HAF": HAF,
    "MyBert": MyBertQA,
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
