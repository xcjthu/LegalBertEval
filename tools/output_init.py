from .output_tool import basic_output_function, null_output_function, ljp_output_function, ms_ljp_output_function
from .output_tool import prf
output_function_dic = {
    "Basic": basic_output_function,
    "Null": null_output_function,
    "LJP": ljp_output_function,
    "ms_LJP": ms_ljp_output_function,
    "prf": prf
}


def init_output_function(config, *args, **params):
    name = config.get("output", "output_function")

    if name in output_function_dic:
        return output_function_dic[name]
    else:
        raise NotImplementedError
