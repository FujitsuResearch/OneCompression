from . import opt
from . import bloom
from . import llama

MODEL_REGISTRY = {'opt': opt.OPT, 'bloom': bloom.BLOOM, 'llama': llama.LLAMA}


def get_model(model_name):
    if 'opt' in model_name:
        return MODEL_REGISTRY['opt']
    elif 'bloom' in model_name:
        return MODEL_REGISTRY['bloom']
    elif 'llama' in model_name:
        return MODEL_REGISTRY['llama']
    return MODEL_REGISTRY[model_name]
