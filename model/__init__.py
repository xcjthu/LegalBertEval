from model.scm.Bert import SCMBert


model_list = {
    "SCMBert": SCMBert,
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
