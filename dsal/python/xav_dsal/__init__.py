import torch
import os
import importlib
from .extension import load_ext
from .registry import register_op, get_all_ops, get_mmcv_patch_ops

try:
    load_ext()
except Exception as e:
    raise RuntimeError(f"failed to load xav_dsal lib, \nerror info: {e}")

_ops_dir = os.path.join(os.path.dirname(__file__), "register_ops")
for filename in os.listdir(_ops_dir):
    if filename.endswith(".py") and not filename.startswith("__"):
        module_name = f"xav_dsal.register_ops.{filename[:-3]}"
        importlib.import_module(module_name)

for name, info in get_all_ops().items():
    globals()[name] = info["func"]

_mmcv_patch_ops = get_mmcv_patch_ops()

def _patch_mmcv(mmcv_ext):
    for func in _mmcv_patch_ops:
        func_name = func.__name__
        mmcv_ext.__dict__[func_name] = func

    return mmcv_ext

__all__ = [
    "boxes_iou3d",
    "nms3d",
    "nms3d_normal",
    "boxes_iou_bev",
    "paired_boxes_iou3d_gpu"
    "dynamic_scatter",
    "roiaware_pool3d",
    "roipoint_pool3d",
    "scatter_add",
    "scatter_max",
    "scatter_min",
    "scatter_mean",
    "scatter_mul",
    "scatter_sum",
    "scatter",
    "bev_pool_v2",
    "voxelization",
    "multi_scale_deform_attn",
    "modulated_deform_conv2d"
    "SparseConv2d",
    "SparseConv3d",
    "SubMConv2d",
    "SubMConv3d",
    "SparseInverseConv2d",
    "SparseInverseConv3d",
    "SparseConvTensor",
    "SparseModule",
    "SparseSequential",

]

from .ops.boxes_iou3d import boxes_iou3d, nms3d, nms3d_normal, boxes_iou_bev, paired_boxes_iou3d_gpu
from .ops.dynamic_scatter import dynamic_scatter
from .ops.roiaware_pool3d import roiaware_pool3d
from .ops.roipoint_pool3d import roipoint_pool3d
from .ops.scatter_add import scatter_add
from .ops.scatter_max import scatter_max
from .ops.scatter_min import scatter_min
from .ops.scatter_mean import scatter_mean
from .ops.scatter_mul import scatter_mul
from .ops.scatter_sum import scatter_sum
from .ops.scatter import scatter
from .ops.bev_pool_v2 import bev_pool_v2
from .ops.voxelize import voxelization
from .ops.multi_scale_deform_attn import multi_scale_deform_attn
from .ops.modulated_deform_conv import modulated_deform_conv2d
from .ops.sparse_conv import (
    SparseConv2d,
    SparseConv3d,
    SubMConv2d,
    SubMConv3d,
    SparseInverseConv2d,
    SparseInverseConv3d,
)

from .ops.sparse_modules import (
    SparseConvTensor,
    SparseModule,
    SparseSequential,
)
