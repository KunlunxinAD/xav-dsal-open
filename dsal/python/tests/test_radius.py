import pytest
import torch
import os
import pandas as pd
import numpy as np

import xav_dsal

def radius(x, y, ptr_x, ptr_y, r, max_num_neighbors, num_workers):
    return xav_dsal.radius(x, y, ptr_x, ptr_y, r, max_num_neighbors, num_workers)

def compare_group_detail(x, y):
    assert x.shape == y.shape and x.shape[0] == 2

    # 排序后方便按 row0 分组
    idx_x = torch.argsort(x[0] * (10**9) + x[1])  # 简单组合键
    idx_y = torch.argsort(y[0] * (10**9) + y[1])

    x_sorted = x[:, idx_x]
    y_sorted = y[:, idx_y]

    # 按 row0 unique 遍历
    keys = torch.unique(x_sorted[0])
    all_ok = True
    for k in keys.tolist():
        mask_x = x_sorted[0] == k
        mask_y = y_sorted[0] == k

        v1 = x_sorted[1][mask_x]
        v2 = y_sorted[1][mask_y]

        if v1.numel() != v2.numel() or not torch.equal(torch.sort(v1).values,
                                                      torch.sort(v2).values):
            print(f"row0 == {k}: mismatch!")
            all_ok = False
    return all_ok

def random_test(x_size, y_size, batch_size, channel, r, max_num_neighbors, num_workers):
    print("!! in random test")
    x = torch.rand((x_size, channel)).to(torch.float32)
    y = torch.rand((y_size, channel)).to(torch.float32)
    if batch_size == 1:
        ptr_x = torch.tensor([0, x_size])
        ptr_y = torch.tensor([0, y_size])

    x_cuda = x.to('cuda').detach()
    y_cuda = y.to('cuda').detach()
    ptr_x_cuda = ptr_x.to('cuda').detach()
    ptr_y_cuda = ptr_y.to('cuda').detach()

    x_cpu = x.to('cpu').detach()
    y_cpu = y.to('cpu').detach()
    ptr_x_cpu = ptr_x.to('cpu').detach()
    ptr_y_cpu = ptr_y.to('cpu').detach()

    output_cpu = radius(x_cpu, y_cpu, ptr_x_cpu, ptr_y_cpu, r, max_num_neighbors, num_workers)
    output_cuda = radius(x_cuda, y_cuda, ptr_x_cuda, ptr_y_cuda, r, max_num_neighbors, num_workers)
    output_cuda = output_cuda.to("cpu")

    assert compare_group_detail(output_cpu, output_cuda)

@pytest.fixture
def base_path():
    return './data/test_radius/'

@pytest.mark.parametrize("group_idx", range(1, 11))
def test_radius(base_path, group_idx):
    data_file = f'{base_path}radius_test{group_idx}.pt'
    assert os.path.exists(data_file), f"Test file {data_file} not exist"

    data = torch.load(data_file, weights_only=True)
    x = data['x']
    y = data['y']
    ptr_x = data['ptr_x']
    ptr_y = data['ptr_y']
    r = data['r']
    max_num_neighbors = data['max_num_neighbors']
    num_workers = data['num_workers']

    print(ptr_x)
    print(ptr_y)

    print(f"x: shape {x.shape}, type {x.dtype}, device {x.device}")
    print(f"y: shape {y.shape}, type {y.dtype}, device {y.device}")
    print(f"ptr_x: shape {ptr_x.shape}, type {ptr_x.dtype}, device {ptr_x.device}")
    print(f"ptr_y: shape {ptr_y.shape}, type {ptr_y.dtype}, device {ptr_y.device}")
    print(f"r {r}, type {type(r)}")
    print(f"max_num_neighbors {max_num_neighbors}, type {type(max_num_neighbors)}")
    print(f"num_workers {num_workers}, type {type(num_workers)}")

    output_cpu = radius(x.to("cpu").detach(), y.to("cpu").detach(), ptr_x.to("cpu").detach(), ptr_y.to("cpu").detach(), r, max_num_neighbors, num_workers)
    output_cuda = radius(x.to("cuda").detach(), y.to("cuda").detach(), ptr_x.to("cuda").detach(), ptr_y.to("cuda").detach(), r, max_num_neighbors, num_workers)
    output_cuda = output_cuda.to("cpu")

    # print(output_cpu.shape)
    # print(output_cpu)
    # print(output_cuda.shape)
    # print(output_cuda)
    # df = pd.DataFrame(output_cpu.T.numpy())
    # df.to_csv("output_cpu.csv", index=False)
    # df = pd.DataFrame(output_cuda.T.numpy())
    # df.to_csv("output_cuda.csv", index=False)

    assert compare_group_detail(output_cpu, output_cuda)
    


if __name__ == "__main__":
    pytest.main(["-v", "-s", "test_radius.py"])
    # random_test(189, 200, 1, 2, 150.0, 301, 1)