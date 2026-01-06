import torch
from ..registry import register_op

try:
    # spconv indice conv
    indice_conv_fp32 = torch.ops.xav_dsal.indice_conv_fp32
    indice_conv_backward_fp32 = torch.ops.xav_dsal.indice_conv_backward_fp32
    indice_conv_half = torch.ops.xav_dsal.indice_conv_half
    indice_conv_backward_half = torch.ops.xav_dsal.indice_conv_backward_half
    register_op("indice_conv_fp32", indice_conv_fp32, for_mmcv=True)
    register_op("indice_conv_backward_fp32", indice_conv_backward_fp32, for_mmcv=True)
    register_op("indice_conv_half", indice_conv_half, for_mmcv=True)
    register_op("indice_conv_backward_half", indice_conv_backward_half, for_mmcv=True)

    # spconv test gather scatter_add
    spconv_gather_fp32 = torch.ops.xav_dsal.spconv_gather_fp32
    spconv_scatter_add_fp32 = torch.ops.xav_dsal.spconv_scatter_add_fp32
    spconv_gather_fp16 = torch.ops.xav_dsal.spconv_gather_fp16
    spconv_scatter_add_fp16 = torch.ops.xav_dsal.spconv_scatter_add_fp16
    register_op("spconv_gather_fp32", spconv_gather_fp32, for_mmcv=True)
    register_op("spconv_scatter_add_fp32", spconv_scatter_add_fp32, for_mmcv=True)
    register_op("spconv_gather_fp16", spconv_gather_fp16, for_mmcv=True)
    register_op("spconv_scatter_add_fp16", spconv_scatter_add_fp16, for_mmcv=True)

except Exception as e:
    raise RuntimeError(f"Failed to load op: {e}")