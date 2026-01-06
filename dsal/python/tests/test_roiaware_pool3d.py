import pytest
import torch
import os
import pandas as pd
import numpy as np
import math

import xav_dsal

from data_compare import get_diff, compare_tensors

class TestRoIAwarePool3dForward:
    def lidar_to_local_coords(self, shift_x, shift_y, rz):
        cosa = math.cos(-rz)
        sina = math.sin(-rz)
        local_x = shift_x * cosa + shift_y * (-sina)
        local_y = shift_x * sina + shift_y * cosa
        
        return local_x, local_y

    def check_pt_in_box3d(self, point, box3d, eps=1e-5):
        x, y, z = np.float32(point[:3])
        cx, cy, cz, dx, dy, dz, rz = np.float32(box3d)
        cz += np.float32(dz / 2.0) # shift to the center since cz in box3d is the bottom center

        cosa = np.float32(np.cos(-rz))
        sina = np.float32(np.sin(-rz))

        local_x = np.float32((x - cx) * cosa + (y - cy) * (-sina))
        local_y = np.float32((x - cx) * sina + (y - cy) * cosa)
        local_z = np.float32(z - cz)

        half_dx = np.float32(dx / 2.0)
        half_dy = np.float32(dy / 2.0)
        half_dz = np.float32(dz / 2.0)

        MODE = 0 # 0:严格模式，1：宽松模式
        if MODE == 1:
            half_dx = np.float32(dx / 2.0 + eps)
            half_dy = np.float32(dy / 2.0 + eps)
            half_dz = np.float32(dz / 2.0 + eps)
        
        if (abs(z - cz) > half_dz):
            return 0, 0, 0
        
        if (local_x > -half_dx) and (local_x < half_dx) and (local_y > -half_dy) and (local_y < half_dy):
            return 1, local_x, local_y
        else:
            return 0, local_x, local_y

    def roiaware_pool3d_golden(self, rois, pts, pts_feature, out, max_pts_per_voxel, mode):
        num_rois = rois.shape[0]
        num_channels = pts_feature.shape[-1]
        num_pts = pts.shape[0]
        
        pooled_features = np.zeros((num_rois, out[0], out[1], out[2], num_channels))
        argmax = np.zeros(shape=(num_rois, out[0], out[1], out[2], num_channels), dtype=int)
        pts_idx_of_voxels = np.zeros(shape=(num_rois, out[0], out[1], out[2], max_pts_per_voxel), dtype=int)
        
        pts_mask = np.ones(shape=(num_rois, num_pts), dtype=int)
        for i in range(num_pts):
            for j in range(num_rois):
                cur_in_flag, local_x, local_y = self.check_pt_in_box3d(pts[i, :], rois[j, :])
                pts_mask[j, i] = -1
                if(cur_in_flag > 0):
                    local_z = pts[i, 2] - rois[j, 2]
                    x_size = rois[j, 3]
                    y_size = rois[j, 4]
                    z_size = rois[j, 5]
                    
                    x_res = x_size / out[0]
                    y_res = y_size / out[1]
                    z_res = z_size / out[2]
                    
                    x_idx = int((local_x + x_size / 2) / x_res)
                    y_idx = int((local_y + y_size / 2) / y_res)
                    z_idx = int(local_z / z_res)
                    indx_encoding = (x_idx << 16) + (y_idx << 8) + z_idx
                    pts_mask[j, i] = indx_encoding

        decoder = 0xFF
        for i in range(num_rois):
            for j in range(num_pts):
                max_num_pts = max_pts_per_voxel - 1
                if(pts_mask[i, j] != -1):
                    idx_encoding = pts_mask[i, j]
                    x_idx = (idx_encoding >> 16) & decoder
                    y_idx = (idx_encoding >> 8) & decoder
                    z_idx = idx_encoding & decoder
                    
                    x_idx = min(max(x_idx, 0), out[0] - 1)
                    y_idx = min(max(y_idx, 0), out[1] - 1)
                    z_idx = min(max(z_idx, 0), out[2] - 1)

                    cnt = pts_idx_of_voxels[i, x_idx, y_idx, z_idx, 0]
                    if(cnt < max_num_pts):
                        pts_idx_of_voxels[i, x_idx, y_idx, z_idx, 0 + cnt + 1] = j
                        pts_idx_of_voxels[i, x_idx, y_idx, z_idx, 0] += 1

        if(mode == 'max'):
            for i in range(out[0]):
                for j in range(out[1]):
                    for k in range(out[2]):
                        for box_idx in range(num_rois):
                            for c_idx in range(num_channels):
                                argmax_idx = -1
                                max_val = -1e10
                                total_pts = pts_idx_of_voxels[box_idx, i, j, k, 0]
                                for p_idx in range(1, total_pts + 1):
                                    if(pts_feature[pts_idx_of_voxels[box_idx, i, j, k, p_idx], c_idx] > max_val):
                                        max_val = pts_feature[pts_idx_of_voxels[box_idx, i, j, k, p_idx], c_idx]
                                        argmax_idx = pts_idx_of_voxels[box_idx, i, j, k, p_idx]
                                        
                                if(argmax_idx != -1):
                                    pooled_features[box_idx, i, j, k, c_idx] = max_val
                                argmax[box_idx, i, j, k, c_idx] = argmax_idx    
        elif(mode == 'avg'):
            for i in range(out[0]):
                for j in range(out[1]):
                    for k in range(out[2]):
                        for box_idx in range(num_rois):
                            for c_idx in range(num_channels):
                                sum_val = 0
                                total_pts = pts_idx_of_voxels[box_idx, i, j, k, 0]
                                for p_idx in range(1, total_pts + 1):
                                    sum_val += pts_feature[pts_idx_of_voxels[box_idx, i, j, k, p_idx], c_idx]
                                    
                                if(total_pts > 0):
                                    pooled_features[box_idx, i, j, k, c_idx] = sum_val / total_pts
        return pooled_features

    def roiaware_pool3d_py(self, rois, pts, pts_feature, out, max_pts_per_voxel, pool_method, dtype):
            # cast
            if (dtype == np.float16):
                rois_cast = rois.astype(np.float32)
                pts_cast = pts.astype(np.float32)
                pts_feature_cast = pts_feature.astype(np.float32)
            elif(dtype == np.float32):
                rois_cast = rois
                pts_cast = pts
                pts_feature_cast = pts_feature

            # Compute
            pooled_features_cpu = self.roiaware_pool3d_golden(rois_cast, pts_cast, pts_feature_cast, out, 
                                                              max_pts_per_voxel, pool_method)

            # cast
            if (dtype == np.float16):
                pooled_features_cpu_cast = pooled_features_cpu.astype(np.float16)
            else:
                pooled_features_cpu_cast = pooled_features_cpu.astype(np.float32)
            return pooled_features_cpu_cast

    def gen_input_data(self, boxes_num, channels, npoints, dtype):
        xyz_coor = np.random.uniform(-1, 1, size=(boxes_num, 3)).astype(dtype)
        xyz_size_num = np.random.uniform(1, 50, size=(1, 3)).astype(dtype)
        xyz_size = (xyz_size_num * np.ones((boxes_num, 3))).astype(dtype)
        angle = np.radians(np.random.randint(0, 360, size=(boxes_num, 1))).astype(dtype)

        rois = np.concatenate((xyz_coor, xyz_size), axis=1)
        rois = np.concatenate((rois, angle), axis=1)

        pts = np.random.uniform(-2, 4, size=(npoints, 3)).astype(dtype)
        pts_feature = np.random.uniform(-1, 1, size=(npoints, channels)).astype(dtype)
        
        return rois, pts, pts_feature

    def one_case(self, boxes_num, out_size, channels, npoints, max_pts_per_voxel, pool_method, dtype):
        rois, pts, pts_feature = self.gen_input_data(boxes_num, channels, npoints, dtype)


        pooled_features_py = self.roiaware_pool3d_py(rois, pts, pts_feature, out_size, max_pts_per_voxel, 
                                                    pool_method, dtype)
        pooled_features_py = torch.from_numpy(pooled_features_py)

        rois_tensor = torch.from_numpy(rois)
        pts_tensor = torch.from_numpy(pts)
        pts_feature_tensor = torch.from_numpy(pts_feature)

        pool_method_map = {'max': 0, 'avg': 1}
        pool_method = pool_method_map[pool_method]

        pooled_features_cuda = xav_dsal.roiaware_pool3d(rois_tensor.to("cuda"), pts_tensor.to("cuda"), 
                                        pts_feature_tensor.to("cuda"), out_size, max_pts_per_voxel, pool_method)
        pooled_features_cuda = pooled_features_cuda.to("cpu")
        res_rel_error, res_abs_error = get_diff(dtype)
        ret = compare_tensors(pooled_features_cuda, pooled_features_py, res_rel_error, res_abs_error)
        assert ret

@pytest.mark.parametrize("boxes_num", [1, 16, 32])
@pytest.mark.parametrize("channels", [4, 16])
@pytest.mark.parametrize("npoints", [16, 32])
@pytest.mark.parametrize("pool_method", ["max", "avg"])
@pytest.mark.parametrize("dtype", [np.float32])
def test_roiaware_pool3d(boxes_num, channels, npoints, pool_method, dtype):
    # boxes_num, out_size, channels, npoints, max_pts_per_voxel, pool_method, dtype
    out_size = (4, 4, 4)
    roiaware_pool3d_fwd = TestRoIAwarePool3dForward()
    roiaware_pool3d_fwd.one_case(boxes_num, out_size, channels, npoints, 8, pool_method, dtype)


class TestRoIAwarePool3dBackward:
    def roiaware_pool3d_grad_cpu(self, pts_idx_of_voxels, argmax, grad_out, npoints, pool_method):
        channels = grad_out.shape[-1]
        grad_in = torch.zeros((npoints, channels)).type_as(grad_out)
        
        # cast
        dtype = grad_out.dtype
        if (dtype == torch.float16):
            grad_out_cast = grad_out.type(torch.float32)
            grad_in_cast = grad_in.type(torch.float32)
        else:
            grad_out_cast = grad_out
            grad_in_cast = grad_in

        # Compute
        if pool_method == 0:
            self.roiaware_maxpool3d_grad_golden(argmax, grad_out_cast, grad_in_cast)
            
        elif pool_method == 1:
            self.roiaware_avgpool3d_grad_golden(pts_idx_of_voxels, grad_out_cast, grad_in_cast)

        # cast
        if (dtype == torch.float16):
            grad_out_cast = grad_out_cast.type(torch.float16)
            grad_in_cast = grad_in_cast.type(torch.float16)
        else:
            grad_out_cast = grad_out
            grad_in_cast = grad_in
        
        return grad_in_cast
    
    def roiaware_maxpool3d_grad_golden(self, argmax, grad_out, grad_in):
        boxes_num, out_x, out_y, out_z, channels = grad_out.shape

        for b in range(boxes_num):
            for ox in range(out_x):
                for oy in range(out_y):
                    for oz in range(out_z):
                        N_idx = argmax[b, ox, oy, oz, :]
                        C_idx = np.arange(channels)
                        grad_in[N_idx, C_idx] += grad_out[b, ox, oy, oz, C_idx]
    
    def roiaware_avgpool3d_grad_golden(self, pts_idx_of_voxels, grad_out, grad_in):
        boxes_num, out_x, out_y, out_z, channels = grad_out.shape
        max_pts_per_voxel = pts_idx_of_voxels.shape[-1]

        for b in range(boxes_num):
            for ox in range(out_x):
                for oy in range(out_y):
                    for oz in range(out_z):
                        total_pts = pts_idx_of_voxels[b, ox, oy, oz, 0]
                        for i in range(1, total_pts + 1):
                            pts_idx = pts_idx_of_voxels[b, ox, oy, oz, i]
                            grad_in[pts_idx, :] += grad_out[b, ox, oy, oz, :] / max(total_pts, 1.0)
    
    def gen_input_data(self, pts_idx_of_voxels_shape, channels, npoints, dtype):
        boxes_num, out_x, out_y, out_z, max_pts_per_voxel = pts_idx_of_voxels_shape
        grad_out = np.random.uniform(-5, 5, (boxes_num, out_x, out_y, out_z, channels)).astype(dtype)
        argmax = np.random.randint(0, npoints, (boxes_num, out_x, out_y, out_z, channels)).astype("int32")
        pts_idx_of_voxels = self.gen_pts_idx_of_voxels(pts_idx_of_voxels_shape, npoints).astype("int32")
        
        grad_out = torch.from_numpy(grad_out)
        argmax = torch.from_numpy(argmax)
        pts_idx_of_voxels = torch.from_numpy(pts_idx_of_voxels)
        return argmax, grad_out, pts_idx_of_voxels

    def gen_pts_idx_of_voxels(self, pts_idx_of_voxels_shape, npoints):
        boxes_num, out_x, out_y, out_z, max_pts_per_voxel = pts_idx_of_voxels_shape
        pts_idx_of_voxels = np.zeros((boxes_num, out_x, out_y, out_z, max_pts_per_voxel - 1)).astype("int32")
        total_pts_array = np.random.randint(0, max_pts_per_voxel, (boxes_num, out_x, out_y, out_z))
        for b in range(boxes_num):
            for ox in range(out_x):
                for oy in range(out_y):
                    for oz in range(out_z):
                        total_pts = total_pts_array[b, ox, oy, oz]
                        choiced_idx = np.array(np.random.choice(npoints, total_pts, replace=False)).astype("int32")
                        choiced_idx = np.sort(choiced_idx)
                        pts_idx_of_voxels[b, ox, oy, oz, 0:total_pts] = choiced_idx
        pts_idx_of_voxels = np.concatenate([total_pts_array.reshape(boxes_num, out_x, out_y, out_z, 1),
                                             pts_idx_of_voxels], axis=-1)
        return pts_idx_of_voxels
    
    def one_case(self, boxes_num, out_size, channels, npoints, max_pts_per_voxel, pool_method, dtype):
        out_x, out_y, out_z = out_size
        pts_idx_of_voxels_shape = (boxes_num, out_x, out_y, out_z, max_pts_per_voxel)
        argmax, grad_out, pts_idx_of_voxels = self.gen_input_data(pts_idx_of_voxels_shape, channels, npoints, dtype)

        golden_grad_in = np.zeros((npoints, channels)).astype(dtype)
        golden_grad_in = torch.from_numpy(golden_grad_in)
        golden_grad_in = self.roiaware_pool3d_grad_cpu(pts_idx_of_voxels, argmax, grad_out, npoints, pool_method)

        res_grad_in = np.zeros((npoints, channels)).astype(dtype)
        res_grad_in = torch.from_numpy(res_grad_in)

        pts_idx_of_voxels = pts_idx_of_voxels.to("cuda")
        argmax = argmax.to("cuda")
        grad_out = grad_out.to("cuda")
        res_grad_in = res_grad_in.to("cuda")
        xav_dsal.roiaware_pool3d_backward(pts_idx_of_voxels, argmax, grad_out, res_grad_in, pool_method)

        res_grad_in = res_grad_in.to("cpu")
        res_rel_error, res_abs_error = get_diff(dtype)
        res_abs_error = 3e-4
        ret = compare_tensors(res_grad_in, golden_grad_in, res_rel_error, res_abs_error)
        assert ret

@pytest.mark.parametrize("boxes_num", [10])
@pytest.mark.parametrize("channels", [256])
@pytest.mark.parametrize("npoints", [128])
@pytest.mark.parametrize("max_pts_per_voxel", [128])
@pytest.mark.parametrize("pool_method", [0, 1])
@pytest.mark.parametrize("dtype", [np.float32])
def test_roiaware_pool3d_backward(boxes_num, channels, npoints, max_pts_per_voxel, pool_method, dtype):
    # boxes_num, out_size, channels, npoints, max_pts_per_voxel, pool_method, dtype
    out_size = (14, 14, 14)
    roiaware_pool3d_bwd = TestRoIAwarePool3dBackward()
    roiaware_pool3d_bwd.one_case(boxes_num, out_size, channels, npoints, max_pts_per_voxel, pool_method, dtype)


if __name__ == "__main__":
    pytest.main(["-v", "-s", "test_roiaware_pool3d.py"])
