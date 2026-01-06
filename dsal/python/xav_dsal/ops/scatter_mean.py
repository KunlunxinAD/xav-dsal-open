from typing import Any, Tuple, Union, Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import Function


def scatter_mean(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
    offset: Optional[Tensor] = None,
    offset_reduce: Optional[int] = None) -> Tensor:
    """Scatter mean operation.
    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The indices tensor.
        dim (int, optional): The axis along which to index. Default: -1.
        out (torch.Tensor or None, optional): The output tensor. Default: None.
        dim_size (int or None, optional): The size of the output tensor along
            the specified dimension. Default: None.
        offset (torch.Tensor or None, optional): The offset tensor. Default: None.
        offset_reduce (int or None, optional): The offset reduce tensor.
            Default: None.
    Returns:
        tuple: The result tensor after scatter mean operation.
    """
    return torch.ops.xav_dsal.scatter_mean(src, index, dim, out, dim_size, offset, offset_reduce)