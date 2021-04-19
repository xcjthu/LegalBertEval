from model.ljp.MultiTaskLJP import MultiTaskLJP

model_list = {
    "MultiTaskLJP": MultiTaskLJP,
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
