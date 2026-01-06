
# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Optional, no_type_check

import torch
import torch.nn as nn
from torch.autograd.function import Function, once_differentiable


class MultiScaleDeformableAttnFunction(Function):

    @staticmethod
    def forward(ctx, value: torch.Tensor, value_spatial_shapes: torch.Tensor,
                value_level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor,
                im2col_step: torch.Tensor) -> torch.Tensor:
        """GPU/MLU version of multi-scale deformable attention.

        Args:
            value (torch.Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (torch.Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (torch.Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (torch.Tensor): The weight of sampling points
                used when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (torch.Tensor): The step used in image to column.

        Returns:
            torch.Tensor: has shape (bs, num_queries, embed_dims)
        """

        ctx.im2col_step = im2col_step

        # When pytorch version >= 1.6.0, amp is adopted for fp16 mode;
        # amp won't cast the type of sampling_locations, attention_weights
        # (float32), but "value" is cast to float16, leading to the type
        # mismatch with input (when it is float32) or weight.
        # The flag for whether to use fp16 or amp is the type of "value",
        # we cast sampling_locations and attention_weights to
        # temporarily support fp16 and amp whatever the
        # pytorch version is.
        sampling_locations = sampling_locations.type_as(value)
        attention_weights = attention_weights.type_as(value)

        output = torch.ops.xav_dsal.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """GPU/MLU version of backward function.

        Args:
            grad_output (torch.Tensor): Gradient of output tensor of forward.

        Returns:
            tuple[Tensor]: Gradient of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        torch.ops.xav_dsal.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None


multi_scale_deform_attn = MultiScaleDeformableAttnFunction.apply
