# Copyright (c) OpenMMLab. All rights reserved.
import os
import numpy as np
import pytest
import torch
from torch.nn.modules.utils import _pair
from typing import Any, List, Tuple, Union

from mmcv.utils import IS_CUDA_AVAILABLE
from data_compare import get_diff, compare_tensors
from data_cache import golden_data_cache, load_data

import xav_dsal as ext_module


class _Voxelization(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx: Any,
            points: torch.Tensor,
            voxel_size: Union[tuple, float],
            coors_range: Union[tuple, float],
            max_points: int = 35,
            max_voxels: int = 20000,
            deterministic: bool = True,
            isMMCV: bool = False) -> Union[Tuple[torch.Tensor], Tuple]:
        """Convert kitti points(N, >=3) to voxels.

        Args:
            points (torch.Tensor): [N, ndim]. Points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity.
            voxel_size (tuple or float): The size of voxel with the shape of
                [3].
            coors_range (tuple or float): The coordinate range of voxel with
                the shape of [6].
            max_points (int, optional): maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize. Default: 35.
            max_voxels (int, optional): maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.
                Default: 20000.
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.

        Returns:
            tuple[torch.Tensor]: tuple[torch.Tensor]: A tuple contains three
            elements. The first one is the output voxels with the shape of
            [M, max_points, n_dim], which only contain points and returned
            when max_points != -1. The second is the voxel coordinates with
            shape of [M, 3]. The last is number of point per voxel with the
            shape of [M], which only returned when max_points != -1.
        """
        if max_points == -1 or max_voxels == -1:
            coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int)
            ext_module.dynamic_voxelize_forward(
                points,
                torch.tensor(voxel_size, dtype=torch.float),
                torch.tensor(coors_range, dtype=torch.float),
                coors,
                NDim=3)
            return coors
        else:
            voxels = points.new_zeros(
                size=(max_voxels, max_points, points.size(1)))
            coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
            num_points_per_voxel = points.new_zeros(
                size=(max_voxels, ), dtype=torch.int)
            if isMMCV:
                voxel_num = torch.zeros(size=(), dtype=torch.long)
                ext_module.hard_voxelize_forward(
                    points,
                    torch.tensor(voxel_size, dtype=torch.float),
                    torch.tensor(coors_range, dtype=torch.float),
                    voxels,
                    coors,
                    num_points_per_voxel,
                    voxel_num,
                    max_points=max_points,
                    max_voxels=max_voxels,
                    NDim=3,
                    deterministic=deterministic)
                # select the valid voxels
                voxels_out = voxels[:voxel_num]
                coors_out = coors[:voxel_num]
                num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
                return voxels_out, coors_out, num_points_per_voxel_out
            else:
                voxel_num = ext_module.hard_voxelize(
                    points,
                    voxels,
                    coors,
                    num_points_per_voxel,
                    voxel_size,
                    coors_range,
                    max_points,
                    max_voxels,
                    3,
                    deterministic,
                )
                # select the valid voxels
                voxels_out = voxels[:voxel_num]
                coors_out = coors[:voxel_num]
                num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
                return voxels_out, coors_out, num_points_per_voxel_out


voxelization = _Voxelization.apply


class Voxelization(torch.nn.Module):
    """Convert kitti points(N, >=3) to voxels.

    Please refer to `Point-Voxel CNN for Efficient 3D Deep Learning
    <https://arxiv.org/abs/1907.03739>`_ for more details.

    Args:
        voxel_size (tuple or float): The size of voxel with the shape of [3].
        point_cloud_range (tuple or float): The coordinate range of voxel with
            the shape of [6].
        max_num_points (int): maximum points contained in a voxel. if
            max_points=-1, it means using dynamic_voxelize.
        max_voxels (int, optional): maximum voxels this function create.
            for second, 20000 is a good choice. Users should shuffle points
            before call this function because max_voxels may drop points.
            Default: 20000.
    """

    def __init__(self,
                 voxel_size: List,
                 point_cloud_range: List,
                 max_num_points: int,
                 max_voxels: Union[tuple, int] = 20000,
                 deterministic: bool = True):
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        """
        super().__init__()

        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
        self.deterministic = deterministic

        point_cloud_range = torch.tensor(
            point_cloud_range, dtype=torch.float32)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (
            point_cloud_range[3:] -  # type: ignore
            point_cloud_range[:3]) / voxel_size  # type: ignore
        grid_size = torch.round(grid_size).long()
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w]
        self.pcd_shape = [*input_feat_shape, 1][::-1]

    def forward(self, input: torch.Tensor, isMMCV: bool) -> torch.Tensor:
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        return voxelization(input, self.voxel_size, self.point_cloud_range,
                            self.max_num_points, max_voxels,
                            self.deterministic, isMMCV)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'voxel_size=' + str(self.voxel_size)
        s += ', point_cloud_range=' + str(self.point_cloud_range)
        s += ', max_num_points=' + str(self.max_num_points)
        s += ', max_voxels=' + str(self.max_voxels)
        s += ', deterministic=' + str(self.deterministic)
        s += ')'
        return s

def _get_voxel_points_indices(points, coors, voxel):
    result_form = np.equal(coors, voxel)
    return result_form[:, 0] & result_form[:, 1] & result_form[:, 2]


@pytest.mark.parametrize('device_type', [
    'cpu',
    pytest.param(
        'cuda:0',
        marks=pytest.mark.skipif(
            not IS_CUDA_AVAILABLE, reason='requires CUDA support'))
])
def test_voxelization(device_type):
    max_num_points = 128
    voxel_size = [0.5, 0.5, 0.5]
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'data/test_voxel.npy')
    voxel_dict = np.load(file_path, allow_pickle=True).item()

    ## !!! the order of voxels and coors
    expected_voxels = voxel_dict['coors'][:,:max_num_points,:]
    expected_coors = voxel_dict['voxels'][:,::-1]
    expected_num_points_per_voxel = voxel_dict['num_points_per_voxel']
    points = voxel_dict['points']
    print("points.shape", points.shape)
    print("expected_coors.shape", expected_coors.shape)
    print("expected_voxels.shape", expected_voxels.shape)
    print("expected_num_points_per_voxel.shape", expected_num_points_per_voxel.shape)

    points = torch.tensor(points)
    hard_voxelization = Voxelization(voxel_size, point_cloud_range,
                                     max_num_points)

    device = torch.device(device_type)

    # test hard_voxelization on cpu/gpu
    points = points.contiguous().to(device)
    voxels, coors, num_points_per_voxel = hard_voxelization.forward(points, False) # FIXME: mmcv test changes position of coors & voxels
    coors = coors.cpu().detach().numpy()
    voxels = voxels.cpu().detach().numpy()
    num_points_per_voxel = num_points_per_voxel.cpu().detach().numpy()

    np.testing.assert_allclose(coors, expected_coors)
    np.testing.assert_allclose(voxels, expected_voxels)
    np.testing.assert_allclose(num_points_per_voxel, expected_num_points_per_voxel)

    # assert np.all(voxels == expected_voxels)
    # assert np.all(coors == expected_coors)
    # assert np.all(num_points_per_voxel == expected_num_points_per_voxel)

# @golden_data_cache(__file__, refresh_data=True)
def gen(point_num, features, dtype, duplicate_rate = 0.3):
        base_num = int(point_num * (1 - duplicate_rate))
        if base_num <= 0:
            base_num = 1
        # generate based point
        x = 100 * np.random.rand(base_num).astype(dtype) - 50 
        y = 100 * np.random.rand(base_num).astype(dtype) - 50 
        z = 8 * np.random.rand(base_num).astype(dtype) - 4
        coors = np.stack([x, y, z], axis=-1)
        feature = np.random.rand(base_num, features - 3).astype(dtype)
        base_points = np.concatenate([coors, feature], axis=-1)
        # generate duplicated point
        duplicate_num = point_num - base_num
        duplicate_indices = np.random.choice(base_num, size=duplicate_num, replace=True)
        duplicate_points = base_points[duplicate_indices]

        all_points = np.concatenate([base_points, duplicate_points], axis=0)
        np.random.shuffle(all_points)

        all_points = torch.from_numpy(all_points)
        return all_points

@pytest.mark.parametrize("point_num", [80])
@pytest.mark.parametrize("features", [4])
@pytest.mark.parametrize("max_num_points", [-1])
@pytest.mark.parametrize("max_voxels", [-1])
@pytest.mark.parametrize("dtype", [np.float32])
def test_dynamic_voxelization(point_num, features, max_num_points, max_voxels, dtype):
    seed = 1024
    np.random.seed(seed)
    voxel_size = [0.2500, 0.2500, 8.0000]
    point_cloud_range = [-50, -50, -4, 50, 50, 4]
    points_cpu = gen(point_num, features, dtype)
    # save_path = os.path.dirname(os.path.abspath(__file__)) + "/data_cache1/test_voxelization/gen/"
    # file_names = ["0194f37aa9_5a94026d84_798e908e9a_.pth"]
    # points_list = load_data(save_path, file_names)
    # points_cpu = points_list[0]
    # print(points_cpu[0:6, :])
    # print("Loaded points shape:", points_cpu.shape)

    hard_voxelization = Voxelization(voxel_size, point_cloud_range,
                                     max_num_points, max_voxels)
    points_xpu = points_cpu.to('cuda')

    coors_cpu = hard_voxelization.forward(points_cpu, True)
    coors_xpu = hard_voxelization.forward(points_xpu, True)
    coors_xpu = coors_xpu.cpu()

    res_rel_error, res_abs_error = get_diff(dtype)
    ret = compare_tensors(coors_cpu, coors_xpu, res_rel_error, res_abs_error)
    assert ret

@pytest.mark.parametrize("point_num", [32])
@pytest.mark.parametrize("features", [4])
@pytest.mark.parametrize("max_num_points", [5])
@pytest.mark.parametrize("max_voxels", [25])
@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize("isMMCV", [False, True])
def test_voxelization_xpu(point_num, features, max_num_points, max_voxels, dtype, isMMCV):
    seed = 1024
    np.random.seed(seed)
    voxel_size = [0.2500, 0.2500, 8.0000]
    point_cloud_range = [-50, -50, -4, 50, 50, 4]
    points_cpu = gen(point_num, features, dtype)
    # save_path = os.path.dirname(os.path.abspath(__file__)) + "/data_cache1/test_voxelization/gen/"
    # file_names = ["0194f37aa9_5a94026d84_798e908e9a_.pth"]
    # points_list = load_data(save_path, file_names)
    # points_cpu = points_list[0]
    # print(points_cpu[0:6, :])
    # print("Loaded points shape:", points_cpu.shape)

    hard_voxelization = Voxelization(voxel_size, point_cloud_range,
                                     max_num_points, max_voxels)
    points_xpu = points_cpu.to('cuda')

    voxels_cpu, coors_cpu, num_points_per_voxel_cpu = hard_voxelization.forward(points_cpu, isMMCV)
    voxels_xpu, coors_xpu, num_points_per_voxel_xpu = hard_voxelization.forward(points_xpu, isMMCV)
    # print("voxels_xpu:", voxels_xpu)
    voxels_xpu = voxels_xpu.cpu()
    coors_xpu = coors_xpu.cpu()
    num_points_per_voxel_xpu = num_points_per_voxel_xpu.cpu()

    res_rel_error, res_abs_error = get_diff(dtype)
    ret1 = compare_tensors(voxels_cpu, voxels_xpu, res_rel_error, res_abs_error)
    assert ret1
    ret2 = compare_tensors(coors_cpu, coors_xpu, res_rel_error, res_abs_error)
    assert ret2
    ret3 = compare_tensors(num_points_per_voxel_cpu, num_points_per_voxel_xpu, res_rel_error, res_abs_error)
    assert ret3


if __name__ == "__main__" :
    test_voxelization('cpu')
    test_voxelization('cuda:0')
    pytest.main(["-v", "-s", "test_voxelization.py"])