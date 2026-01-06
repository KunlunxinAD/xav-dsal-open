import torch
import pytest
import xav_dsal
import numpy as np
from data_compare import compare_tensors
from torch.profiler import profile, record_function, ProfilerActivity


def batch_line_interpolate(batch_points, num_points):
    """
    批量 LineString 插值

    Args:
        batch_points: tensor of shape (batch_size, n, 2)
        num_points: points to interpolate per line

    Returns:
        tensor of shape (batch_size, num_points, 2)
    """
    batch_size, n, _ = batch_points.shape

    # 计算每条线的累积距离
    diffs = batch_points[:, 1:] - batch_points[:, :-1]        # 计算每个点和上一个点的x y差
    segment_lengths = torch.norm(diffs, dim=2)                # 计算L2欧式距离，即根号(x^2+y^2)
    cumulative_dists = torch.cat([
        torch.zeros(batch_size, 1, device=batch_points.device),    # 第一个点距离0
        torch.cumsum(segment_lengths, dim=1)                       # 每个点和第一个点的距离
    ], dim=1)
    total_lengths = cumulative_dists[:, -1]

    # 生成目标距离
    target_dists = torch.zeros(batch_size, num_points, device=batch_points.device)
    for i in range(batch_size):
        target_dists[i] = torch.linspace(0, total_lengths[i], num_points, device=batch_points.device) # 获取0到线段总长的num_points个均匀分布

    # 批量处理
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_points)
    seg_indices = torch.searchsorted(cumulative_dists, target_dists) - 1
    seg_indices = torch.clamp(seg_indices, 0, n - 2)

    seg_starts = cumulative_dists[batch_indices, seg_indices]
    seg_ends = cumulative_dists[batch_indices, seg_indices + 1]
    t_vals = (target_dists - seg_starts) / (seg_ends - seg_starts + 1e-8)

    start_pts = batch_points[batch_indices, seg_indices]
    end_pts = batch_points[batch_indices, seg_indices + 1]
    return start_pts + t_vals.unsqueeze(2) * (end_pts - start_pts)

@pytest.mark.parametrize("batch_size", range(1, 11))
def test_linear_interpolate(batch_size):
# def test_linear_interpolate():
    read_pt = False
    save_pt = False
    test_id = 0
    # batch_size = 4
    # n = 5
    n = np.random.randint(2, 10)
    num_points = 20

    if read_pt:
        batch_points = torch.load(f"./data/test_linear_interpolate/batch_points_{test_id}.pt", weights_only=True)
    else:
        batch_points = torch.rand(batch_size, n, 2, dtype=torch.float32)
        if save_pt:
            torch.save(batch_points, f"./data/test_linear_interpolate/batch_points_{test_id}.pt")

    # print("batch_points", batch_points)
    # print("num_points", num_points)
    
    output_python = batch_line_interpolate(batch_points, num_points)

    # batch_points_cpu = batch_points.to("cpu")
    # output_cpu = xav_dsal.linear_interpolate(batch_points_cpu, num_points)

    batch_points_cuda = batch_points.to("cuda")
    output_cuda = xav_dsal.linear_interpolate(batch_points_cuda, num_points)
    output_cuda = output_cuda.to("cpu")

    # print("output_python", output_python)
    # # print("output_cpu", output_cpu)
    # print("output_cuda", output_cuda)

    ret = compare_tensors(output_cuda, output_python, 0.0, 3e-5)
    # ret = compare_tensors(output_cuda, output_cpu, 0.0, 3e-5)
    assert ret
    

if __name__ == "__main__":
    pytest.main(["-v", "-s", "test_linear_interpolate.py"])