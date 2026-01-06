import importlib
import builtins
import sys, os
import warnings

if not hasattr(importlib, "__origin__import_module__"):
    importlib.__origin__import_module__ = importlib.import_module


def _custom_import_module(module_name, package=None):
    if module_name == "mmcv._ext":
        import xav_dsal
        mmcv_ext = importlib.__origin__import_module__(module_name, package)
        return xav_dsal._patch_mmcv(mmcv_ext)
    else:
        return importlib.__origin__import_module__(module_name, package)


def hook():
    disable_automatic_xav = int(os.environ.get("DISABLE_XAV_AUTO", "0"))
    if disable_automatic_xav:
        return
    importlib.import_module = _custom_import_module

