from typing import Any, Tuple, Union, Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import Function


def scatter(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
    offset: Optional[Tensor] = None,
    offset_reduce: Optional[int] = None):
    """Scatter operation with various reduction methods.
    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The indices tensor.
        dim (int, optional): The axis along which to index. Default: -1.
        out (torch.Tensor or None, optional): The output tensor. Default: None.
        dim_size (int or None, optional): The size of the output tensor along
            the specified dimension. Default: None.
        reduce (str, optional): The reduction method to apply. Options are
            "sum", "mean", "min", "max", "mul". Default: "sum".
        offset (torch.Tensor or None, optional): The offset tensor. Default: None.
        offset_reduce (int or None, optional): The offset reduce tensor.
            Default: None.
    Returns:
        torch.Tensor: The result tensor after scatter operation.
    """
    return torch.ops.xav_dsal.scatter(src, index, dim, out, dim_size, reduce,
                                      offset, offset_reduce)