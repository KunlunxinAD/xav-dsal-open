import torch
from ..registry import register_op

try:
    # hard_voxelize for bev version
    hard_voxelize = torch.ops.xav_dsal.hard_voxelize
    register_op("hard_voxelize", hard_voxelize, for_mmcv=True)
    # hard_voxelize_forward for MMCV version
    hard_voxelize_forward = torch.ops.xav_dsal.hard_voxelize_forward
    register_op("hard_voxelize_forward", hard_voxelize_forward, for_mmcv=True)
    # dynamic_voxelize
    dynamic_voxelize_forward = torch.ops.xav_dsal.dynamic_voxelize_forward
    register_op("dynamic_voxelize_forward", dynamic_voxelize_forward, for_mmcv=True)

except Exception as e:
    raise RuntimeError(f"Failed to load op: {e}")