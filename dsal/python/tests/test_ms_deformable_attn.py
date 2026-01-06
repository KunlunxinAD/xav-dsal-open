import unittest

import numpy as np
import pytest
import torch
import copy
import os
import torch.nn.functional as F
from torch.autograd.function import (
    Function,
    once_differentiable,
)

# import ms_deformable_attn_ext
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    (
        bs,
        _,
        num_heads,
        embed_dims,
    ) = value.shape
    (
        _,
        num_queries,
        num_heads,
        num_levels,
        num_points,
        _,
    ) = sampling_locations.shape
    value_list = value.split(
        [H_ * W_ for H_, W_ in value_spatial_shapes],
        dim=1,
    )
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for (
        level,
        (
            H_,
            W_,
        ),
    ) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(
                1,
                2,
            )
            .reshape(
                bs * num_heads,
                embed_dims,
                H_,
                W_,
            )
        )
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = (
            sampling_grids[
                :,
                :,
                :,
                level,
            ]
            .transpose(
                1,
                2,
            )
            .flatten(
                0,
                1,
            )
        )
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(
        1,
        2,
    ).reshape(
        bs * num_heads,
        1,
        num_queries,
        num_levels * num_points,
    )
    output = (
        (
            torch.stack(
                sampling_value_list,
                dim=-2,
            ).flatten(-2)
            * attention_weights
        )
        .sum(-1)
        .view(
            bs,
            num_heads * embed_dims,
            num_queries,
        )
    )
    return output.transpose(
        1,
        2,
    ).contiguous()


class MultiScaleDeformableAttnFunction(Function):
    @staticmethod
    def forward(
        ctx,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        value_level_start_index: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        im2col_step: torch.Tensor,
    ) -> torch.Tensor:
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

        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)

        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple:
        """GPU/MLU version of backward function.

        Args:
            grad_output (torch.Tensor): Gradient of output tensor of forward.

        Returns:
            tuple[Tensor]: Gradient of input tensors in forward.
        """
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
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

        return (
            grad_value,
            None,
            None,
            grad_sampling_loc,
            grad_attn_weight,
            None,
        )


class TestCustomMMCVMsDeformAttn(unittest.TestCase):
    def func_test_forward_equal_with_pytorch_float(
        test_case,
        device,
    ):
        (batch_size, num_heads, num_channels) = (6, 8, 32)
        # (num_querys, num_levels, num_points) = (200, 1, 8)
        # shapes = torch.as_tensor([(20, 20)], dtype=torch.long)
        (num_querys, num_levels, num_points) = (200, 4, 13)
        shapes = torch.as_tensor([(20, 20), (10, 10), (5, 5), (3, 3)], dtype=torch.long)
        level_start_index = torch.cat(
            (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
        )
        spatial_size = sum((H * W).item() for H, W in shapes)

        torch.manual_seed(3)
        value = torch.rand(batch_size, spatial_size, num_heads, num_channels) * 0.01
        sampling_locations = torch.rand(
            batch_size, num_querys, num_heads, num_levels, num_points, 2
        )
        attention_weights = (
            torch.rand(batch_size, num_querys, num_heads, num_levels, num_points) + 1e-5
        )
        attention_weights /= attention_weights.sum(-1, keepdim=True).sum(
            -2, keepdim=True
        )
        im2col_step = 2
        output_pytorch = (
            multi_scale_deformable_attn_pytorch(
                value, shapes, sampling_locations, attention_weights
            )
            .detach()
            .cpu()
        )

        output_cuda = (
            MultiScaleDeformableAttnFunction.apply(
                value.to(device=device),
                shapes.to(device=device),
                level_start_index.to(device=device),
                sampling_locations.to(device=device),
                attention_weights.to(device=device),
                im2col_step,
            )
            .detach()
            .cpu()
        )

        max_abs_err = (output_cuda - output_pytorch).abs().max()
        max_rel_err = ( (output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()
        test_case.assertTrue(max_abs_err < 1e-4)
        test_case.assertTrue(max_rel_err < 1e-4)
        print(f">> max_abs_err={max_abs_err}, max_rel_err={max_rel_err}")
        test_case.assertTrue(torch.allclose(output_cuda, output_pytorch, rtol=1e-5, atol=1e-5))

    def test_forward_equal_with_pytorch_float(test_case,):
        test_case.func_test_forward_equal_with_pytorch_float("cpu")
        test_case.func_test_forward_equal_with_pytorch_float("cuda")

    def func_test_backward_equal_with_pytorch_float(
        test_case,
        device,
    ):
        (batch_size, num_heads, num_channels) = (6, 80, 32)
        # (num_querys, num_levels, num_points) = (200, 1, 8)
        # shapes = torch.as_tensor([(20, 20)], dtype=torch.long)
        (num_querys, num_levels, num_points) = (200, 4, 13)
        shapes = torch.as_tensor([(20, 20), (10, 10), (5, 5), (3, 3)], dtype=torch.long)
        level_start_index = torch.cat(
            (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
        )
        spatial_size = sum((H * W).item() for H, W in shapes)

        torch.manual_seed(3)
        value = torch.rand(batch_size, spatial_size, num_heads, num_channels) * 0.01
        sampling_locations = torch.rand(
            batch_size, num_querys, num_heads, num_levels, num_points, 2
        )
        attention_weights = (
            torch.rand(batch_size, num_querys, num_heads, num_levels, num_points) + 1e-5
        )
        attention_weights /= attention_weights.sum(-1, keepdim=True).sum(
            -2, keepdim=True
        )
        im2col_step = 2

        value.requires_grad = True
        sampling_locations.requires_grad = True
        attention_weights.requires_grad = True
        value_cuda = copy.deepcopy(value).to(device=device)
        shapes_cuda = copy.deepcopy(shapes).to(device=device)
        level_start_index_cuda = copy.deepcopy(level_start_index).to(device=device)
        sampling_locations_cuda = copy.deepcopy(sampling_locations).to(device=device)
        attention_weights_cuda = copy.deepcopy(attention_weights).to(device=device)

        im2col_step = 2

        output_pytorch = multi_scale_deformable_attn_pytorch(
            value, shapes, sampling_locations, attention_weights
        )
        grad_output_pytorch = 2.0 * torch.rand_like(output_pytorch) - 1.0
        # grad_output_pytorch = torch.rand_like(output_pytorch)
        grad_output_cuda = copy.deepcopy(grad_output_pytorch).to(device=device)

        output_pytorch.backward(grad_output_pytorch)
        grad_value = value.grad.detach().cpu()
        grad_location = sampling_locations.grad.detach().cpu()
        grad_attn_weight = attention_weights.grad.detach().cpu()

        output_cuda = MultiScaleDeformableAttnFunction.apply(
            value_cuda,
            shapes_cuda,
            level_start_index_cuda,
            sampling_locations_cuda,
            attention_weights_cuda,
            im2col_step,
        )
        value_cuda.retain_grad()
        sampling_locations_cuda.retain_grad()
        attention_weights_cuda.retain_grad()

        output_cuda.backward(grad_output_cuda)
        grad_value_cuda = value_cuda.grad.detach().cpu()
        grad_location_cuda = sampling_locations_cuda.grad.detach().cpu()
        grad_attn_weight_cuda = attention_weights_cuda.grad.detach().cpu()

        test_case.assertTrue(
            torch.allclose(
                grad_value_cuda,
                grad_value,
                atol=1e-3,
                rtol=1e-3,
            )
        )

        diff_grad_loc = grad_location_cuda - grad_location
        max_abs_err = diff_grad_loc.abs().max()
        max_rel_err = ( diff_grad_loc.abs() / grad_location.abs()).max()
        print(f"grad_location>> max_abs_err={max_abs_err}, max_rel_err={max_rel_err}, mean={diff_grad_loc.abs().mean()}")

        test_case.assertTrue(
            torch.allclose(
                grad_location_cuda,
                grad_location,
                atol=1e-3,
                rtol=1e-3,
            )
        )
        test_case.assertTrue(
            torch.allclose(
                grad_attn_weight_cuda,
                grad_attn_weight,
                atol=1e-3,
                rtol=1e-3,
            )
        )

    def test_backward_equal_with_pytorch_float(
        test_case,
    ):
        test_case.func_test_backward_equal_with_pytorch_float("cpu")
        test_case.func_test_backward_equal_with_pytorch_float("cuda")


if __name__ == "__main__":
    unittest.main()
