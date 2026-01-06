import pytest
import torch
import random
import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple

from data_compare import compare_tensors, compare_int_tensors
from test_unique_dim import generate_self_with_duplicates
import xav_dsal

run_counter = -1


@pytest.fixture
def base_path():
    return "./data/test_dynamic_scatter/"


@pytest.mark.parametrize("test_id", range(8))
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
@pytest.mark.parametrize("reduce", ["mean", "max", "sum"])
def test_dynamic_scatter(base_path, test_id, dtype, reduce):
    read_pt = False
    save_pt = False
    global run_counter
    run_counter += 1

    coors_shape = [30, 100, 5000, 10000, 50000, 100000, 900000, 360003]
    valid_lines = [7, 88, 4888, 9900, 48888, 98888, 888888, 1]
    C = 30

    if read_pt:
        data = torch.load(f"{base_path}dynamic_scatter_input_{run_counter}.pt", weights_only=True)
        reduce = data["reduce"]
    else:
        feat = torch.randn(coors_shape[test_id], C)  # [N, C] 特征
        coors = generate_self_with_duplicates(coors_shape[test_id], 3, valid_lines[test_id], dtype)
        data = {"feat": feat.to("cpu"), "coors": coors.to("cpu"), "reduce": reduce}
        # coors = torch.randint(0, 100, (10000, 3))  # [N, 3] 体素坐标
        if save_pt:
            torch.save(data, f"{base_path}dynamic_scatter_input_{run_counter}.pt")

    print("------test fwd")
    print("reduce: ", reduce)
    print("feat shape: ", data["feat"].shape)
    print("coors shape: ", data["coors"].shape)

    feat_cpu = data["feat"].to("cpu").detach()
    coors_cpu = data["coors"].to("cpu").detach()
    feat_cuda = data["feat"].to("cuda").detach()
    coors_cuda = data["coors"].to("cuda").detach()

    (voxel_feats_cpu, voxel_coors_cpu, point2voxel_map_cpu, voxel_points_count_cpu) = xav_dsal.dynamic_scatter_forward(
        feat_cpu, coors_cpu, reduce
    )
    (voxel_feats_cuda, voxel_coors_cuda, point2voxel_map_cuda, voxel_points_count_cuda) = (
        xav_dsal.dynamic_scatter_forward(feat_cuda, coors_cuda, reduce)
    )

    assert compare_int_tensors(point2voxel_map_cpu, point2voxel_map_cuda.to("cpu"), "dynamic_scatter point2voxel_map")
    assert compare_int_tensors(
        voxel_points_count_cpu, voxel_points_count_cuda.to("cpu"), "dynamic_scatter voxel_points_count"
    )
    assert compare_int_tensors(voxel_coors_cpu, voxel_coors_cuda.to("cpu"), "dynamic_scatter voxel_coors")
    assert compare_tensors(voxel_feats_cpu, voxel_feats_cuda.to("cpu"), 5e-3, 5e-3, "dynamic_scatter voxel_feats")
    # if save_pt:
    #     out_data = {
    #         "voxel_feats": voxel_feats_cpu,
    #         "voxel_coors": voxel_coors_cpu,
    #         "point2voxel_map": point2voxel_map_cpu,
    #         "voxel_points_count": voxel_points_count_cpu,
    #     }
    #     torch.save(out_data, f"{base_path}dynamic_scatter_output_{run_counter}.pt")

    print("\n\n------test bwd")
    print("voxel_feats", voxel_feats_cpu.shape, voxel_feats_cpu.dtype)
    print("point2voxel_map", point2voxel_map_cpu.shape, point2voxel_map_cpu.dtype)
    print("voxel_points_count", voxel_points_count_cpu.shape, voxel_points_count_cpu.dtype)

    grad_feats_cpu = torch.zeros_like(feat_cpu, device="cpu")
    grad_feats_cuda = torch.zeros_like(feat_cuda, device="cuda")
    grad_voxel_feats = torch.randn_like(voxel_feats_cpu)

    xav_dsal.dynamic_scatter_backward(
        grad_feats_cpu,
        grad_voxel_feats.to("cpu"),
        feat_cpu,
        voxel_feats_cpu,
        point2voxel_map_cpu,
        voxel_points_count_cpu,
        reduce,
    )

    xav_dsal.dynamic_scatter_backward(
        grad_feats_cuda,
        grad_voxel_feats.to("cuda"),
        feat_cuda,
        voxel_feats_cuda,
        point2voxel_map_cuda,
        voxel_points_count_cuda,
        reduce,
    )

    assert compare_tensors(grad_feats_cpu, grad_feats_cuda.to("cpu"), 5e-5, 5e-5, "dynamic_scatter grad_feats")


if __name__ == "__main__":
    pytest.main(["-v", "-s", "test_dynamic_scatter.py"])
