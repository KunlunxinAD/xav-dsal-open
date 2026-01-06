import torch
from typing import Tuple, Type

import numpy as np


def get_diff(dtype: Type[np.number]) -> Tuple[float, float]:
    if dtype == np.float32:
        res_rel_error = 0.0
        res_abs_error = 3e-5
    elif dtype == np.float16:
        res_rel_error = 0.0
        res_abs_error = 2e-3
    elif dtype == np.dtype("bfloat16"):
        res_rel_error = 8e-3
        res_abs_error = 5e-5
    else:
        raise TypeError(f"Unsupported type: {dtype}")

    return res_rel_error, res_abs_error


def compare_tensors(a, b, rtol, atol, name=""):
    if name:
        print(f"-- compare tensor {name}")
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    print(f"tensor shape is: {tuple(a.shape)}")
    # 计算每个元素的误差是否满足条件
    is_close = torch.isclose(a, b, rtol=rtol, atol=atol)

    # overall判断是否全部满足
    all_close = torch.all(is_close)
    print(f"All values within tolerance: {all_close.item()}")

    # 计算误差值
    diff = torch.abs(a - b)

    # 找出误差超出范围的位置
    error_mask = ~is_close
    error_indices = torch.nonzero(error_mask, as_tuple=False)

    if error_indices.numel() == 0:
        print("No differences exceeding tolerance.")
        return True

    # 限制最多打印 50 个索引和对应误差
    max_display = 50
    num_to_display = min(max_display, error_indices.size(0))

    print(f"\nFirst {num_to_display} indices with errors beyond tolerance:")
    for i in range(num_to_display):
        idx = tuple(error_indices[i].tolist())
        print(f"Index {idx}: a={a[idx].item()}, b={b[idx].item()}, diff={diff[idx].item()}")
    return False


def compare_int_tensors(out_cpu, out_cuda, op_name="unknown_op"):
    """
    对比整数类型的 CPU 张量和 CUDA 张量, 返回一致性结果 + 具体差异细节

    参数:
        out_cpu:  CPU 整数张量（如 torch.int32, torch.int64)
        out_cuda: CUDA 整数张量（需与 out_cpu 同 shape、同 dtype)
        op_name:  操作名（用于打印日志, 如 "aten.scatter.default")

    返回:
        all_okay: 布尔值, True 表示完全一致, False 表示存在差异
        msg:      字符串, 包含差异细节（无差异则提示一致)
    """
    if op_name:
        print(f"-- 对比张量 {op_name}")
    if out_cpu.shape != out_cuda.shape:
        all_okay = False
        print(f"[{op_name}] 形状不匹配: A shape={out_cpu.shape}, B shape={out_cuda.shape}")
        return all_okay

    if out_cpu.dtype != out_cuda.dtype:
        all_okay = False
        print(f"[{op_name}] dtype 不匹配: A dtype={out_cpu.dtype}, B dtype={out_cuda.dtype}")
        return all_okay

    print(f"形状: {tuple(out_cpu.shape)}")

    out_cpu = out_cpu.cpu()
    out_cuda = out_cuda.cpu()
    equal_mask = torch.eq(out_cpu, out_cuda)
    unequal_indices = torch.nonzero(~equal_mask, as_tuple=False)

    if unequal_indices.numel() == 0:
        all_okay = True
        msg = f"[{op_name}] 整数张量比较: CPU 和 CUDA 结果一致 (shape={out_cpu.shape}, dtype={out_cpu.dtype})"
    else:
        all_okay = False
        num_unequal = unequal_indices.shape[0]  # 正确的差异数量
        max_show = min(10, num_unequal)  # 最多显示 10 处差异
        msg = f"[{op_name}] 整数张量比较: CPU 和 CUDA 结果不一致！共 {num_unequal} 处差异, 前 {max_show} 处如下: \n"

        for i in range(max_show):
            idx = unequal_indices[i]
            cpu_val = out_cpu[tuple(idx)]
            cuda_val = out_cuda[tuple(idx)]
            msg += f"  索引 {tuple(idx)}: CPU={cpu_val}, CUDA={cuda_val}\n"

        if num_unequal > max_show:
            msg += f"  ...（省略剩余 {num_unequal - max_show} 处差异)"

    print(msg)
    return all_okay
