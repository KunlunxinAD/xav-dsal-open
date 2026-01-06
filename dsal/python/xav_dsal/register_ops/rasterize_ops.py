import torch
from ..registry import register_op

try:
    # rasterization
    forward_rasterize = torch.ops.xav_dsal.forward_rasterize
    backward_rasterize = torch.ops.xav_dsal.backward_rasterize
    register_op("forward_rasterize", forward_rasterize, for_mmcv=True)
    register_op("backward_rasterize", backward_rasterize, for_mmcv=True)

    forward_rasterize_xtrans = torch.ops.xav_dsal.forward_rasterize_xtrans
    backward_rasterize_xtrans = torch.ops.xav_dsal.backward_rasterize_xtrans
    register_op("forward_rasterize_xtrans", forward_rasterize_xtrans, for_mmcv=True)
    register_op("backward_rasterize_xtrans", backward_rasterize_xtrans, for_mmcv=True)
except Exception as e:
    raise RuntimeError(f"Failed to load op: {e}")