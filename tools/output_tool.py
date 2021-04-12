import json

from .accuracy_tool import gen_micro_macro_result


def null_output_function(data, config, *args, **params):
    return ""


def basic_output_function(data, config, *args, **params):
    temp = gen_micro_macro_result(data)
    result = []
    for name in ["mip"]:
        result.append(temp[name])

    return json.dumps(result, sort_keys=True)

def binary_output_function(data, config, *args, **params):
    if data['total'] == 0:
        return json.dumps({'acc': 0})
    else:
        return json.dumps({'acc': round(data['right'] / data['total'], 4)})
