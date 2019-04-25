import os, sys
import importlib
import torch as tc

def load_model(params):
    spec = importlib.util.spec_from_file_location("model", params.model_def_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = getattr(module, params.model_name)()
    model.load_state_dict(tc.load(params.model_path))
    return model.cuda()