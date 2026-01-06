import pytest
import torch
import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from data_compare import compare_tensors
from data_compare import compare_int_tensors
import xav_dsal


def generate_self_with_duplicates(total_rows=100, cols=3, unique_base=80, self_type=torch.int32):
    """
    生成包含重复行的随机张量
    参数:
        total_rows: 总行数（100）
        cols: 列数（3）
        unique_base: 基础唯一行的数量（控制重复率，值越小重复越多）
    返回:
        self: 形状为 [100, 3] 的张量，包含重复行
    """
    if self_type == torch.int32 or self_type == torch.int64:
        base_rows = torch.randint(low=0, high=20, size=(unique_base, cols), dtype=self_type)
    else:
        base_rows = torch.rand(size=(unique_base, cols), dtype=torch.float32)

    # 2. 从基础行中随机选择并重复，凑满100行
    # 生成100个索引（范围0~unique_base-1），用于从base_rows中选择行
    select_indices = torch.randint(low=0, high=unique_base, size=(total_rows,))
    data = base_rows[select_indices]  # 按索引选择行，产生重复

    # 3. 打乱行顺序（可选，使重复行分布更随机）
    data = data[torch.randperm(total_rows)]

    return data


@pytest.fixture
def base_path():
    return "./data/test_unique_dim/"


@pytest.mark.parametrize("test_id", range(9))
def test_unique_dim(base_path, test_id):
    read_pt = True
    save_pt = False

    # fmt: off
    self_shapes = [[12, 3], [2, 6, 2], [24, 3], [100, 4], [5000, 4], [10000, 4], [30000, 3], [50000, 3], [900000, 4]]
    valid_lines = [7,        4,         19,      88,       4808,      9900,       28910,      46750,      888888]
    dims =        [0,        1,         0,       0,        0,         0,          0,          0,          0]
    # fmt: on

    if read_pt:
        self_data = torch.load(f"{base_path}self_data_{test_id}.pt", weights_only=True)
        dim = torch.load(f"{base_path}dim_{test_id}.pt", weights_only=True)
        output = torch.load(f"{base_path}output_{test_id}.pt", weights_only=True)
        inverse_indices = torch.load(f"{base_path}inverse_indices_{test_id}.pt", weights_only=True)
        counts = torch.load(f"{base_path}counts_{test_id}.pt", weights_only=True)

    else:
        # 编造数据 需要先把dim挪到0维构造两维重复数据，再还原成原本维度
        dim = dims[test_id]
        self_shape = self_shapes[test_id]
        dim_size = 1
        dim_other = 1
        for i in range(len(self_shape)):
            if i == dim:
                dim_size = self_shape[i]
            else:
                dim_other *= self_shape[i]
        print(f"flat shape to generate data: rows {dim_size}, cols {dim_other}")
        self_data = generate_self_with_duplicates(dim_size, dim_other, valid_lines[test_id], torch.int32)
        if dim != 0:
            self_shape_dim0 = [self_shape[dim]] + self_shape[:dim] + self_shape[dim + 1 :]
            self_data = self_data.reshape(self_shape_dim0)
            self_data = self_data.transpose(0, dim).contiguous()

        if save_pt:
            torch.save(self_data, f"{base_path}self_data_{test_id}.pt")
            torch.save(dim, f"{base_path}dim_{test_id}.pt")

        (output, inverse_indices, counts) = xav_dsal.unique_dim(self_data, dim, True, True, True)

        if save_pt:
            torch.save(output, f"{base_path}output_{test_id}.pt")
            torch.save(inverse_indices, f"{base_path}inverse_indices_{test_id}.pt")
            torch.save(counts, f"{base_path}counts_{test_id}.pt")

    self_data_cuda = self_data.to("cuda").detach()
    (output_cuda, inverse_indices_cuda, counts_cuda) = xav_dsal.unique_dim(self_data_cuda, dim, True, True, True)
    output_cuda = output_cuda.to("cpu")
    inverse_indices_cuda = inverse_indices_cuda.to("cpu")
    counts_cuda = counts_cuda.to("cpu")

    # print("self_data", self_data)
    # print("dim", dim)

    # print("output", output)
    # print("inverse_indices", inverse_indices)
    # print("counts", counts)

    # print("output_cuda", output_cuda)
    # print("inverse_indices_cuda", inverse_indices_cuda)
    # print("counts_cuda", counts_cuda)

    assert compare_tensors(output, output_cuda, 1e-5, 1e-5, "unique_dim output")
    assert compare_tensors(counts, counts_cuda, 1e-5, 1e-5, "unique_dim counts")
    assert compare_tensors(inverse_indices, inverse_indices_cuda, 1e-5, 1e-5, "unique_dim inverse_indices")


if __name__ == "__main__":
    pytest.main(["-v", "-s", "test_unique_dim.py"])
