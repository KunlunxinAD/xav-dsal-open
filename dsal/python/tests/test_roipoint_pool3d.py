import pytest
import torch
import numpy as np
import xav_dsal

from data_compare import get_diff, compare_tensors
from data_cache import golden_data_cache, load_data

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=float('inf'))

def debug_print(bs_idx, bs_target, boxes_idx, boxes_target, boxes, points):
    if bs_idx == bs_target and boxes_idx == boxes_target:
        print("boxes\n")
        print(boxes)
        print("points\n")
        print(points)

# @golden_data_cache(__file__, refresh_data=True)
def gen_input_data(batch_size, boxes_num, points_num, channels, dtype = np.float32):
    np.random.seed(0)
    xyz_coor = np.random.uniform(-1, 1, size=(batch_size, boxes_num, 3)).astype(dtype)
    xyz_size = np.random.uniform(1, 50, size=(batch_size, boxes_num, 3)).astype(dtype)
    angle = np.radians(np.random.randint(0, 360, size=(batch_size, boxes_num, 1))).astype(dtype)

    boxes3d = np.concatenate((xyz_coor, xyz_size, angle), axis=2)

    points = np.random.uniform(-2, 4, size=(batch_size, points_num, 3)).astype(dtype)
    points_feature = np.random.uniform(-1, 1, size=(batch_size, points_num, channels)).astype(dtype)

    return boxes3d, points, points_feature

def check_point_in_box3d(point, box3d, eps=1e-5):
    x, y, z = np.float32(point[:3])
    cx, cy, cz, dx, dy, dz, rz = np.float32(box3d)
    cz += np.float32(dz / 2.0)

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
        
    return int(abs(local_x) < half_dx and abs(local_y) < half_dy and abs(local_z) <= half_dz)

def roipoint_pool3d_forward(num_sampled_points, points, point_features, boxes3d, pooled_features, bs_idx, boxes_idx):
    point_num = points.shape[0]  # N
    feature_len = point_features.shape[1]  # C
    point_idx = np.zeros((num_sampled_points), dtype=np.int32)  # (num)
    cnt = 0
    for pt_idx in range(point_num):
        flag = check_point_in_box3d(points[pt_idx], boxes3d)
        if flag == 1:
            point_idx[cnt] = pt_idx
            cnt += 1
        if cnt == num_sampled_points:
            break

    if cnt == 0:
        return 1
    if cnt < num_sampled_points:
        for spn_idx in range(cnt, num_sampled_points):
            point_idx[spn_idx] = point_idx[spn_idx % cnt]

    for sample_point_idx in range(num_sampled_points):
        src_point_idx = point_idx[sample_point_idx]
        pooled_features[sample_point_idx, 0:3] = points[src_point_idx, 0:3]
        pooled_features[sample_point_idx, 3 : 3 + feature_len] = point_features[src_point_idx, 0:feature_len]
    return 0

def cpu_roipoint_pool3d(num_sampled_points, points, point_features, boxes3d):
    # B=batch_size; N=point_num; M=boxes_num; C=feature_len; num = num_sampled_points
    batch_size = points.shape[0]  # B
    feature_len = point_features.shape[2]  # C
    boxes_num = boxes3d.shape[1]  # M
    pooled_features = np.zeros_like(points, shape=(batch_size, boxes_num, num_sampled_points, 3 + feature_len))
    pooled_empty_flag = np.zeros((batch_size, boxes_num), dtype=np.int32)
    for bs_idx in range(batch_size):
        for boxes_idx in range(boxes_num):
            pooled_empty_flag[bs_idx][boxes_idx] = roipoint_pool3d_forward(
                num_sampled_points,
                points[bs_idx],
                point_features[bs_idx],
                boxes3d[bs_idx][boxes_idx],
                pooled_features[bs_idx][boxes_idx],
                bs_idx,
                boxes_idx
            )
    return pooled_features, pooled_empty_flag

class TestRoIPointPool3dForward:
    def one_case(self, batch_size, boxes_num, point_num, num_sampled_points, channels, dtype):
        boxes3d, points, point_features = gen_input_data(batch_size, boxes_num, point_num, channels, dtype)
        # 数据缓存复现
        # file_names = ["018f5c4626_5a47034f12_b99a527996_a29f18e71a_798e908e9a_0.npy", "018f5c4626_5a47034f12_b99a527996_a29f18e71a_798e908e9a_1.npy", "018f5c4626_5a47034f12_b99a527996_a29f18e71a_798e908e9a_2.npy"]
        # boxes3d, points, point_features = load_data("/workspace/baidu/xpu/xav-dsal/dsal/python/tests/data_cache/test_roipoint_pool3d/gen_input_data", file_names)

        # 计算python版本的
        cpu_pooled_features, cpu_pooled_empty_flag = cpu_roipoint_pool3d(
            num_sampled_points, points, point_features, boxes3d
        )

        cpu_pooled_features = torch.from_numpy(cpu_pooled_features)
        cpu_pooled_empty_flag = torch.from_numpy(cpu_pooled_empty_flag)

        pooled_features = np.zeros_like(points, shape=(batch_size, boxes_num, num_sampled_points, 3 + channels))
        pooled_empty_flag = np.zeros((batch_size, boxes_num), dtype=np.int32)

        points_tensor_cpu = torch.from_numpy(points)
        point_features_tensor_cpu = torch.from_numpy(point_features)
        boxes3d_tensor_cpu = torch.from_numpy(boxes3d)
        pooled_features_tensor_cpu = torch.from_numpy(pooled_features)
        pooled_empty_flag_tensor_cpu = torch.from_numpy(pooled_empty_flag)
        if 0:
            # 计算cpp版本的
            xav_dsal.roipoint_pool3d_forward(points_tensor_cpu, boxes3d_tensor_cpu, point_features_tensor_cpu,
                                            pooled_features_tensor_cpu, pooled_empty_flag_tensor_cpu)
            res_rel_error, res_abs_error = get_diff(dtype)
            ret0 = compare_tensors(pooled_empty_flag_tensor_cpu, cpu_pooled_empty_flag, res_rel_error, res_abs_error)
            assert ret0
            ret1 = compare_tensors(pooled_features_tensor_cpu, cpu_pooled_features, res_rel_error, res_abs_error)
            assert ret1

        if 1:
            # 计算xpu版本
            points_tensor_cuda = points_tensor_cpu.to("cuda")
            point_features_tensor_cuda = point_features_tensor_cpu.to("cuda")
            boxes3d_tensor_cuda = boxes3d_tensor_cpu.to("cuda")
            pooled_features_tensor_cuda = pooled_features_tensor_cpu.to("cuda")
            pooled_empty_flag_tensor_cuda = pooled_empty_flag_tensor_cpu.to("cuda")
            xav_dsal.roipoint_pool3d_forward(points_tensor_cuda, boxes3d_tensor_cuda, point_features_tensor_cuda,
                                            pooled_features_tensor_cuda, pooled_empty_flag_tensor_cuda)
            pooled_features_tensor_cuda = pooled_features_tensor_cuda.to("cpu")
            pooled_empty_flag_tensor_cuda = pooled_empty_flag_tensor_cuda.to("cpu")
            res_rel_error, res_abs_error = get_diff(dtype)
            ret0 = compare_tensors(pooled_empty_flag_tensor_cuda, cpu_pooled_empty_flag, res_rel_error, res_abs_error)
            assert ret0
            ret1 = compare_tensors(pooled_features_tensor_cuda, cpu_pooled_features, res_rel_error, res_abs_error)
            assert ret1

# point_num以20480作为分界线，全载或非全载
# channels以509作为分界线，全载或非全载
@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.parametrize("boxes_num", [128])
@pytest.mark.parametrize("point_num", [16384])
@pytest.mark.parametrize("num_sampled_points", [512])
@pytest.mark.parametrize("channels", [131])
@pytest.mark.parametrize("dtype", [np.float32])
def test_roipoint_pool3d_forward(batch_size, boxes_num, point_num, num_sampled_points, channels, dtype):
    roipoint_pool3d_fwd = TestRoIPointPool3dForward()
    roipoint_pool3d_fwd.one_case(batch_size, boxes_num, point_num, num_sampled_points, channels, dtype)

if __name__ == "__main__":
    pytest.main(["-v", "-s", "test_roipoint_pool3d.py"])


