import torch
from ..registry import register_op

try:
    bev_pool_v2_forward = torch.ops.xav_dsal.bev_pool_v2_forward
    bev_pool_v2_backward = torch.ops.xav_dsal.bev_pool_v2_backward
    register_op("bev_pool_v2_forward", bev_pool_v2_forward, for_mmcv=True)
    register_op("bev_pool_v2_backward", bev_pool_v2_backward, for_mmcv=True)

    # roiaware_pool3d
    roiaware_pool3d_forward = torch.ops.xav_dsal.roiaware_pool3d_forward
    roiaware_pool3d_backward = torch.ops.xav_dsal.roiaware_pool3d_grad
    register_op("roiaware_pool3d_forward", roiaware_pool3d_forward, for_mmcv=True)
    register_op("roiaware_pool3d_backward", roiaware_pool3d_backward, for_mmcv=True)

    # roipoint_pool3d_fwd
    roipoint_pool3d_forward = torch.ops.xav_dsal.roipoint_pool3d_forward
    register_op("roipoint_pool3d_forward", roipoint_pool3d_forward, for_mmcv=True)


except Exception as e:
    raise RuntimeError(f"Failed to load op: {e}")