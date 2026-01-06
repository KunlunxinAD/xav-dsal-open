import pytest
import torch
import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from data_compare import compare_tensors
from data_compare import compare_int_tensors

import xav_dsal

# torch.set_printoptions(threshold=torch.inf, linewidth=200)
torch.set_printoptions(sci_mode=False, precision=5)


def scatter_seperate(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
    offset=None,
    offset_reduce=None,
):
    if reduce == "sum" or reduce == "add":
        return xav_dsal.scatter_sum(src, index, dim, out, dim_size, offset, offset_reduce)
    elif reduce == "mul":
        return xav_dsal.scatter_mul(src, index, dim, out, dim_size, offset, offset_reduce)
    elif reduce == "mean":
        return xav_dsal.scatter_mean(src, index, dim, out, dim_size, offset, offset_reduce)
    elif reduce == "min":
        return xav_dsal.scatter_min(src, index, dim, out, dim_size, offset, offset_reduce)
    elif reduce == "max":
        return xav_dsal.scatter_max(src, index, dim, out, dim_size, offset, offset_reduce)
    else:
        raise ValueError


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
    offset=None,
    offset_reduce=None,
):
    print("offset", offset_reduce, offset)
    return xav_dsal.scatter(src, index, dim, out, dim_size, reduce, offset, offset_reduce)


def random_test_scatter(
    src_shape,
    index_shape,
    dim,
    out_shape,
    dim_size,
    reduce,
    i,
    d_type,
    read_pt,
    save_pt,
    folder_name,
    offset=None,
    offset_reduce=None,
):
    index_max = out_shape[dim]
    index_min = 0

    if read_pt:
        print("read pt")
        src = torch.load(f"./data/test_scatter/{folder_name}/src_{i}.pt", weights_only=True).to(d_type)
        out = torch.load(f"./data/test_scatter/{folder_name}/out_{i}.pt", weights_only=True).to(d_type)
        index = torch.load(f"./data/test_scatter/{folder_name}/index_{i}.pt", weights_only=True).to(torch.long)
    else:
        if d_type == torch.float32:
            src = torch.rand(src_shape, dtype=d_type)
            out = torch.rand(out_shape, dtype=d_type)
        else:
            src = torch.randint(low=0, high=500, size=src_shape, dtype=d_type)
            out = torch.randint(low=0, high=500, size=out_shape, dtype=d_type)
        index = torch.randint(low=index_min, high=index_max, size=index_shape, dtype=torch.long)
        index, indices = torch.sort(index)
        if save_pt:
            torch.save(src, f"./data/test_scatter/{folder_name}/src_{i}.pt")
            torch.save(out, f"./data/test_scatter/{folder_name}/out_{i}.pt")
            torch.save(index, f"./data/test_scatter/{folder_name}/index_{i}.pt")

    # print("src", src)
    # print("out", out)
    # print("index", index)

    src_cpu = src.to("cpu").detach()
    out_cpu = out.clone().to("cpu").detach()
    index_cpu = index.to("cpu").detach()

    src_cuda = src.to("cuda").detach()
    out_cuda = out.clone().to("cuda").detach()
    index_cuda = index.to("cuda").detach()

    if reduce == "max" or reduce == "min":
        (out_cpu, arg_out_cpu) = scatter(src_cpu, index_cpu, dim, out_cpu, dim_size, reduce, offset, offset_reduce)
        (out_cuda, arg_out_cuda) = scatter(src_cuda, index_cuda, dim, out_cuda, dim_size, reduce, offset, offset_reduce)
        arg_out_cuda = arg_out_cuda.to("cpu")
    else:
        (out_cpu,) = scatter(src_cpu, index_cpu, dim, out_cpu, dim_size, reduce, offset, offset_reduce)
        (out_cuda,) = scatter(src_cuda, index_cuda, dim, out_cuda, dim_size, reduce, offset, offset_reduce)
    out_cuda = out_cuda.to("cpu")

    # if reduce == "max" or reduce == "min":
    #     print("arg_out_cpu", arg_out_cpu)
    #     print("arg_out_cuda", arg_out_cuda)
    # print("out_cpu", out_cpu)
    # print("out_cuda", out_cuda)

    flag = True
    msg = ""
    if reduce == "max" or reduce == "min":
        all_okay = compare_int_tensors(arg_out_cpu, arg_out_cuda, "scatter_reduce_arg")
        if not all_okay:
            print(msg)
            flag = False

    if d_type == torch.float32:
        all_okay = compare_tensors(out_cpu, out_cuda, 1e-4, 1e-4, "scatter_reduce_out")
    else:
        all_okay = compare_int_tensors(out_cpu, out_cuda, "scatter_reduce_out")
    if not all_okay:
        print(f"*** scatter gpu/xpu output precision test failed; Details: {msg}")
        flag = False

    if flag:
        print(f"*** 通过测试")
    return flag


@pytest.mark.parametrize("reduce_type", ["mul", "sum", "mean"])
@pytest.mark.parametrize("data_type", [torch.float32, torch.int32, torch.int64])
@pytest.mark.parametrize("test_id", range(1))
def test_scatter_sum_mul_mean(reduce_type, data_type, test_id):
    offset = None
    offset_reduce = None
    folder_name = "sum_mul_mean"
    # fmt: off
    ### index.dim = 1, dim = 0
    src_shape = [(20, 5), (2000, 520), (5, 20, 5), (2000, 30, 11)]
    index_shape = [(20,), (2000,), (5,), (2000,)]
    out_shape = [(30, 5), (3250, 520), (10, 20, 5), (3180, 30, 11)]
    dim = [0, 0, 0, 0]
    dim_size = [30, 3250, 10, 3180]

    ### index.dim = 1, dim != 0
    src_shape += [(2, 8), (520, 2100), (20, 8, 5), (30, 2105, 11)]
    index_shape += [(8,), (2100,), (5,), (2105,)]
    out_shape += [(2, 12), (520, 3250), (20, 8, 10), (30, 3180, 11)]
    dim += [1, 1, 2, 1]
    dim_size += [12, 3250, 10, 3180]

    ### dim < index.dim && index.dim != 1
    src_shape += [(5, 10), (33, 10), (100, 350, 52), (100, 64, 52)]
    index_shape += [(5, 10), (33, 10), (100, 350), (100, 64, 52)]
    out_shape += [(8, 10), (33, 25), (100, 440, 52), (100, 64, 70)]
    dim += [0, 1, 1, 2]
    dim_size += [8, 25, 440, 70]

    ### dim > index.dim && index.dim != 1
    src_shape += [(10, 15, 12), (2, 2, 3, 4), (10, 12, 310, 180)]
    index_shape += [(10, 15), (2, 2), (10, 12)]
    out_shape += [(10, 15, 20), (2, 2, 3, 8), (10, 12, 310, 230)]
    dim += [2, 3, 3]
    dim_size += [20, 8, 230]
    # fmt: on

    res = random_test_scatter(
        src_shape[test_id],
        index_shape[test_id],
        dim[test_id],
        out_shape[test_id],
        dim_size[test_id],
        reduce_type,
        test_id,
        data_type,
        True,
        False,
        folder_name,
        offset,
        offset_reduce,
    )
    assert res


@pytest.mark.parametrize("reduce_type", ["max", "min"])
@pytest.mark.parametrize("data_type", [torch.int, torch.int64])
@pytest.mark.parametrize("test_id", range(8))
def test_scatter_max_min(reduce_type, data_type, test_id):
    folder_name = "max_min"
    # fmt: off
    ### index.dim = 1, dim = 0
    src_shape   = [(20, 5), (5, 20, 5)]
    index_shape = [(20,),   (5,)]
    out_shape   = [(30, 5), (10, 20, 5)]
    dim         = [0,       0]
    dim_size    = [30,      10]

    ### index.dim = 1, dim != 0
    src_shape   += [(2, 8),  (10, 8, 5)]
    index_shape += [(8,),    (5,)]
    out_shape   += [(2, 12), (10, 8, 10)]
    dim         += [1,       2]
    dim_size    += [12,      10]

    ### dim < index.dim && index.dim != 1
    src_shape   += [(5, 10), (33, 10)]
    index_shape += [(5, 10), (33, 10)]
    out_shape   += [(8, 10), (33, 25)]
    dim         += [0,       1]
    dim_size    += [8,       25]

    ### dim > index.dim && index.dim != 1
    src_shape   += [(10, 15, 12), (2, 2, 3, 4)]
    index_shape += [(10, 15),     (2, 2)]
    out_shape   += [(10, 15, 20), (2, 2, 3, 8)]
    dim         += [2,            3]
    dim_size    += [20,           8]
    # fmt: on

    res = random_test_scatter(
        src_shape[test_id],
        index_shape[test_id],
        dim[test_id],
        out_shape[test_id],
        dim_size[test_id],
        reduce_type,
        test_id,
        data_type,
        True,
        False,
        folder_name,
    )
    assert res


@pytest.mark.parametrize("reduce_type", ["max", "min", "sum", "mean", "mul"])
@pytest.mark.parametrize("test_id", range(5))
def test_scatter_reduce_one(test_id, reduce_type):
    folder_name = "reduce_one"
    # int32类型的data在数据量特别大的时候重复数很多，和CPU结果有差距
    data_type = torch.float32
    # fmt: off
    src_shape   = [(20, 5), (70, 20), (200, 32), (3000, 32), (360003, 32)]
    index_shape = [(20,),   (70,),    (200,),    (3000,),    (360003,)]
    out_shape   = [(1, 5),  (1, 20),  (1, 32),   (1, 32),    (1, 32)]
    dim         = [0,       0,        0,         0,          0]
    dim_size    = [1,       1,        1,         1,          1]
    # fmt: on

    res = random_test_scatter(
        src_shape[test_id],
        index_shape[test_id],
        dim[test_id],
        out_shape[test_id],
        dim_size[test_id],
        reduce_type,
        test_id,
        data_type,
        False,
        False,
        folder_name,
    )
    assert res


if __name__ == "__main__":
    pytest.main(["-v", "-s", "test_scatter.py"])
