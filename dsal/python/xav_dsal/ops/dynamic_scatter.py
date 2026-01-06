from typing import Any, Tuple, Union, Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import Function


def dynamic_scatter(
    feats: Tensor,
    coors: Tensor,
    reduce_type: str = 'max'):
    """Dynamic scatter operation.

    Args:
        feats (torch.Tensor): The input features tensor.
        coors (torch.Tensor): The coordinates tensor.
        reduce_type (str, optional): The reduction type, can be 'max', 'sum', or
            'mean'. Default: 'max'.

    Returns:
        tuple: A tuple of (voxel_feats, voxel_coors) where voxel_feats is the
            result features tensor and voxel_coors is the coordinates tensor.
    """
    if (torch.numel(feats) == 0 or torch.numel(coors) == 0):
            raise Exception("Error! Input Tensor cannot be an empty tensor.\n")

    if reduce_type not in ("max", "sum", "mean"):
        raise ValueError("reduce_type should be 'max', 'sum' or 'mean', but now is %s." % reduce_type)

    return torch.ops.xav_dsal.dynamic_scatter_forward(feats, coors, reduce_type)