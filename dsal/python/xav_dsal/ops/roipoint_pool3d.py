from typing import Any, Tuple, Union
import torch
import torch.nn as nn
from torch.autograd import Function


class RoIPointPool3dFunction(Function):
    @staticmethod
    def forward(
            ctx: Any,
            points: torch.Tensor,
            point_features: torch.Tensor,
            boxes3d: torch.Tensor,
            num_sampled_points: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points (torch.Tensor): Input points whose shape is (B, N, C).
            point_features (torch.Tensor): Features of input points whose shape
                is (B, N, C).
            boxes3d (B, M, 7), Input bounding boxes whose shape is (B, M, 7).
            num_sampled_points (int, optional): The num of sampled points.
                Default: 512.

        Returns:
            tuple[torch.Tensor]: A tuple contains two elements. The first one
            is the pooled features whose shape is (B, M, 512, 3 + C). The
            second is an empty flag whose shape is (B, M).
        """
        assert len(points.shape) == 3 and points.shape[2] == 3
        batch_size, boxes_num, feature_len = points.shape[0], boxes3d.shape[
            1], point_features.shape[2]
        pooled_boxes3d = boxes3d.view(batch_size, -1, 7)
        pooled_features = point_features.new_zeros(
            (batch_size, boxes_num, num_sampled_points, 3 + feature_len))
        pooled_empty_flag = point_features.new_zeros(
            (batch_size, boxes_num)).int()

        torch.ops.xav_dsal.roipoint_pool3d_forward(points.contiguous(), 
                                         pooled_boxes3d.contiguous(),
                                         point_features.contiguous(),
                                         pooled_features, pooled_empty_flag)

        return pooled_features, pooled_empty_flag

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


roipoint_pool3d = RoIPointPool3dFunction.apply