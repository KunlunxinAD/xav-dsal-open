from typing import Any, Tuple, Union
import torch
import torch.nn as nn
from torch.autograd import Function


class RoIAwarePool3dFunction(Function):
    @staticmethod
    def forward(
        ctx: Any, 
        rois: torch.Tensor, 
        pts: torch.Tensor, 
        pts_feature: torch.Tensor, 
        out_size: Union[int, tuple], 
        max_pts_each_voxel: int, 
        pool_method: int):
        """
        Args:
            ctx:
            rois: (N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
            pts: (npoints, 3)
            pts_feature: (npoints, C)
            out_size: int or tuple, like 7 or (7, 7, 7)
            max_pts_each_voxel:
            pool_method: 'max' 0 or 'avg' 1

        Returns:
            pooled_features: (N, out_x, out_y, out_z, C)
        """
        assert rois.shape[1] == 7 and pts.shape[1] == 3
        if isinstance(out_size, int):
            out_x = out_y = out_z = out_size
        else:
            assert len(out_size) == 3
            for k in range(3):
                assert isinstance(out_size[k], int)
            out_x, out_y, out_z = out_size

        num_rois = rois.shape[0]
        num_channels = pts_feature.shape[-1]
        num_pts = pts.shape[0]

        pooled_features = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels))
        argmax = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels), dtype=torch.int)
        pts_idx_of_voxels = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, max_pts_each_voxel), dtype=torch.int)

        torch.ops.xav_dsal.roiaware_pool3d_forward(rois, pts, pts_feature, argmax, pts_idx_of_voxels, pooled_features, 
                                         pool_method)
        ctx.save_for_backward(pts_idx_of_voxels, argmax)
        ctx.pool_method = pool_method
        ctx.num_pts = num_pts
        ctx.num_channels = num_channels
        return pooled_features

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor):
        """
        :param grad_out: (N, out_x, out_y, out_z, C)
        :return:
            grad_in: (npoints, C)
        """
        pts_idx_of_voxels, argmax = ctx.saved_tensors
        pool_method = ctx.pool_method
        num_pts = ctx.num_pts
        num_channels = ctx.num_channels
        grad_in = grad_out.new_zeros((num_pts, num_channels))
        torch.ops.xav_dsal.roiaware_pool3d_backward(pts_idx_of_voxels, argmax, 
                                                    grad_out.contiguous(), grad_in, pool_method)
        return None, None, grad_in, None, None, None


roiaware_pool3d = RoIAwarePool3dFunction.apply


