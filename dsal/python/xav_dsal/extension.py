import os
import pkgutil
import importlib
import importlib.machinery
import warnings

import torch


def load_ext():
    script_path = os.path.dirname(__file__)
    torch.ops.load_library(f'{script_path}/_ext_xpu.so')


def check_ext_exist() -> bool:
    ext_loader = pkgutil.find_loader('xav_dsal._ext_xpu')
    return ext_loader is not None

