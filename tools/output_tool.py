import json

from .accuracy_tool import gen_micro_macro_result


def null_output_function(data, config, *args, **params):
    return ""

def prf(data, config, *args, **params):
    if (data["TP"] + data["FP"]) == 0:
        precision = 0
    else:
        precision = data["TP"] / (data["TP"] + data["FP"])
    if (data["TP"] + data["FN"]) == 0:
        recall = 0
    else:
        recall = data["TP"] / (data["TP"] + data["FN"])
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}

def basic_output_function(data, config, *args, **params):
    which = config.get("output", "output_value").replace(" ", "").split(",")
    temp = gen_micro_macro_result(data)
    result = {}
    for name in which:
        result[name] = temp[name]

    return json.dumps(result, sort_keys=True)

def ms_ljp_output_function(data, config, *args, **params):
    temp = {}
    temp["charge"] = gen_micro_macro_result(data["charge"])
    temp["law"] = gen_micro_macro_result(data["law"])
    result = {}
    for name in ["charge", "law"]:
        result[name] = {'mif': temp[name]['mif'], 'maf': temp[name]['maf']}

    return json.dumps(result, sort_keys=True)


def ljp_output_function(data, config, *args, **params):
    temp = {}
    temp["charge"] = gen_micro_macro_result(data["charge"])
    temp["law"] = gen_micro_macro_result(data["law"])
    temp["term"] = data["term"]
    result = {}
    for name in ["charge", "law"]:
        result[name] = {'mif': temp[name]['mif'], 'maf': temp[name]['maf']}
        # for name_ in ["mip", "mir", "mif", "map", "mar", "maf"]:
        #     result[name].append(temp[name][name_])

    result["term"] = round(data["term"][1] / data["term"][0], 4)

    return json.dumps(result, sort_keys=True)
