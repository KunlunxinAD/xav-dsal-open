import torch
from ..registry import register_op

try:
    ms_deform_attn_forward = torch.ops.xav_dsal.ms_deform_attn_forward
    ms_deform_attn_backward = torch.ops.xav_dsal.ms_deform_attn_backward
    register_op("ms_deform_attn_forward", ms_deform_attn_forward, for_mmcv=True)
    register_op("ms_deform_attn_backward", ms_deform_attn_backward, for_mmcv=True)

    modulated_deform_conv_forward = torch.ops.xav_dsal.modulated_deform_conv_forward
    modulated_deform_conv_backward = torch.ops.xav_dsal.modulated_deform_conv_backward
    register_op("modulated_deform_conv_forward", modulated_deform_conv_forward, for_mmcv=True)
    register_op("modulated_deform_conv_backward", modulated_deform_conv_backward, for_mmcv=True)

    deformable_aggregation_forward = torch.ops.xav_dsal.deformable_aggregation_forward
    deformable_aggregation_backward = torch.ops.xav_dsal.deformable_aggregation_backward
    register_op("deformable_aggregation_forward", deformable_aggregation_forward, for_mmcv=True)
    register_op("deformable_aggregation_backward", deformable_aggregation_backward, for_mmcv=True)

    dcnv4_forward = torch.ops.xav_dsal.dcnv4_forward
    dcnv4_backward = torch.ops.xav_dsal.dcnv4_backward
    register_op("dcnv4_forward", dcnv4_forward, for_mmcv=True)
    register_op("dcnv4_backward", dcnv4_backward, for_mmcv=True)

    geometric_kernel_attn_cuda_forward = torch.ops.xav_dsal.geometric_kernel_attn_forward
    geometric_kernel_attn_cuda_backward = torch.ops.xav_dsal.geometric_kernel_attn_backward
    register_op("geometric_kernel_attn_cuda_forward", geometric_kernel_attn_cuda_forward, for_mmcv=True)
    register_op("geometric_kernel_attn_cuda_backward", geometric_kernel_attn_cuda_backward, for_mmcv=True)

    local_aggregate_forward = torch.ops.xav_dsal.local_aggregate_forward
    local_aggregate_backward = torch.ops.xav_dsal.local_aggregate_backward
    register_op("local_aggregate_forward", local_aggregate_forward, for_mmcv=True)
    register_op("local_aggregate_backward", local_aggregate_backward, for_mmcv=True)
except Exception as e:
    raise RuntimeError(f"Failed to load op: {e}")