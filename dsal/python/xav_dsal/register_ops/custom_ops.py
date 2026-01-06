import torch
from ..registry import register_op

try:
    # smooth_cosine_loss
    smooth_cosine_loss_forward = torch.ops.xav_dsal.smooth_cosine_loss_forward
    smooth_cosine_loss_backward = torch.ops.xav_dsal.smooth_cosine_loss_backward
    register_op("smooth_cosine_loss_forward", smooth_cosine_loss_forward, for_mmcv=False)
    register_op("smooth_cosine_loss_backward", smooth_cosine_loss_backward, for_mmcv=False)

    # linear_interpolate
    linear_interpolate = torch.ops.xav_dsal.linear_interpolate
    register_op("linear_interpolate", linear_interpolate, for_mmcv=False)

except Exception as e:
    raise RuntimeError(f"Failed to load op: {e}")