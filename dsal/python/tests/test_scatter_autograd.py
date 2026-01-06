from itertools import product

import pytest
import torch
from typing import Optional, Tuple
from torch.autograd import gradcheck

import xav_dsal

tests = [
    {
        'src': [1, 3, 2, 4, 5, 6],
        'index': [0, 1, 0, 1, 1, 3],
        'dim': -1,
        'sum': [3, 12, 0, 6],
        'add': [3, 12, 0, 6],
        'mul': [2, 60, 1, 6],
        'mean': [1.5, 4, 0, 6],
        'min': [1, 3, 0, 6],
        'arg_min': [0, 1, 6, 5],
        'max': [2, 5, 0, 6],
        'arg_max': [2, 4, 6, 5],
    },
    {
        'src': [[1, 2], [5, 6], [3, 4], [7, 8], [9, 10], [11, 12]],
        'index': [0, 1, 0, 1, 1, 3],
        'dim': 0,
        'sum': [[4, 6], [21, 24], [0, 0], [11, 12]],
        'add': [[4, 6], [21, 24], [0, 0], [11, 12]],
        'mul': [[1 * 3, 2 * 4], [5 * 7 * 9, 6 * 8 * 10], [1, 1], [11, 12]],
        'mean': [[2, 3], [7, 8], [0, 0], [11, 12]],
        'min': [[1, 2], [5, 6], [0, 0], [11, 12]],
        'arg_min': [[0, 0], [1, 1], [6, 6], [5, 5]],
        'max': [[3, 4], [9, 10], [0, 0], [11, 12]],
        'arg_max': [[2, 2], [4, 4], [6, 6], [5, 5]],
    },
    {
        'src': [[1, 5, 3, 7, 9, 11], [2, 4, 8, 6, 10, 12]],
        'index': [[0, 1, 0, 1, 1, 3], [0, 0, 1, 0, 1, 2]],
        'dim': 1,
        'sum': [[4, 21, 0, 11], [12, 18, 12, 0]],
        'add': [[4, 21, 0, 11], [12, 18, 12, 0]],
        'mul': [[1 * 3, 5 * 7 * 9, 1, 11], [2 * 4 * 6, 8 * 10, 12, 1]],
        'mean': [[2, 7, 0, 11], [4, 9, 12, 0]],
        'min': [[1, 5, 0, 11], [2, 8, 12, 0]],
        'arg_min': [[0, 1, 6, 5], [0, 2, 5, 6]],
        'max': [[3, 9, 0, 11], [6, 10, 12, 0]],
        'arg_max': [[2, 4, 6, 5], [3, 4, 5, 6]],
    },
    {
        'src': [[[1, 2], [5, 6], [3, 4]], [[10, 11], [7, 9], [12, 13]]],
        'index': [[0, 1, 0], [2, 0, 2]],
        'dim': 1,
        'sum': [[[4, 6], [5, 6], [0, 0]], [[7, 9], [0, 0], [22, 24]]],
        'add': [[[4, 6], [5, 6], [0, 0]], [[7, 9], [0, 0], [22, 24]]],
        'mul': [[[3, 8], [5, 6], [1, 1]], [[7, 9], [1, 1], [120, 11 * 13]]],
        'mean': [[[2, 3], [5, 6], [0, 0]], [[7, 9], [0, 0], [11, 12]]],
        'min': [[[1, 2], [5, 6], [0, 0]], [[7, 9], [0, 0], [10, 11]]],
        'arg_min': [[[0, 0], [1, 1], [3, 3]], [[1, 1], [3, 3], [0, 0]]],
        'max': [[[3, 4], [5, 6], [0, 0]], [[7, 9], [0, 0], [12, 13]]],
        'arg_max': [[[2, 2], [1, 1], [3, 3]], [[1, 1], [3, 3], [2, 2]]],
    },
    {
        'src': [[1, 3], [2, 4]],
        'index': [[0, 0], [0, 0]],
        'dim': 1,
        'sum': [[4], [6]],
        'add': [[4], [6]],
        'mul': [[3], [8]],
        'mean': [[2], [3]],
        'min': [[1], [2]],
        'arg_min': [[0], [0]],
        'max': [[3], [4]],
        'arg_max': [[1], [1]],
    },
    {
        'src': [[[1, 1], [3, 3]], [[2, 2], [4, 4]]],
        'index': [[0, 0], [0, 0]],
        'dim': 1,
        'sum': [[[4, 4]], [[6, 6]]],
        'add': [[[4, 4]], [[6, 6]]],
        'mul': [[[3, 3]], [[8, 8]]],
        'mean': [[[2, 2]], [[3, 3]]],
        'min': [[[1, 1]], [[2, 2]]],
        'arg_min': [[[0, 0]], [[0, 0]]],
        'max': [[[3, 3]], [[4, 4]]],
        'arg_max': [[[1, 1]], [[1, 1]]],
    },
]

@pytest.mark.parametrize("test_id", range(6))
@pytest.mark.parametrize("reduce", ["sum", "mul", "mean", "max", "min"])
@pytest.mark.parametrize("data_type", [torch.float32, torch.int, torch.int64])
def test_forward(test_id, reduce, data_type):

    src_cpu = torch.tensor(tests[test_id]['src'], dtype=data_type, device='cpu')
    src_cuda = torch.tensor(tests[test_id]['src'], dtype=data_type, device='cuda')
    index_cpu = torch.tensor(tests[test_id]['index'], dtype=torch.long, device='cpu')
    index_cuda = torch.tensor(tests[test_id]['index'], dtype=torch.long, device='cuda')
    dim = tests[test_id]['dim']
    expected = torch.tensor(tests[test_id][reduce], dtype=data_type, device='cpu')

    fn = getattr(xav_dsal, 'scatter_' + reduce) 
    out_cpu = fn(src_cpu, index_cpu, dim, None, None, None, None)
    out_cuda = fn(src_cuda, index_cuda, dim, None, None, None, None)

    if isinstance(out_cuda, tuple):
        out_cpu, arg_out_cpu = out_cpu
        out_cuda, arg_out_cuda = out_cuda
        arg_out_cuda = arg_out_cuda.to("cpu")
        arg_expected = torch.tensor(tests[test_id]['arg_' + reduce], dtype=torch.long, device='cpu')
        assert torch.all(arg_out_cuda == arg_expected)
        assert arg_out_cuda.tolist() == arg_expected.tolist()
        assert torch.all(arg_out_cpu == arg_expected)
        assert arg_out_cpu.tolist() == arg_expected.tolist()
    out_cuda = out_cuda.to("cpu")
    assert torch.all(out_cuda == expected)
    assert out_cuda.tolist() == expected.tolist()
    assert torch.all(out_cpu == expected)
    assert out_cpu.tolist() == expected.tolist()

@pytest.mark.parametrize("test_id", range(6))
@pytest.mark.parametrize("reduce", ["sum", "mul", "mean", "max", "min"])
@pytest.mark.parametrize("data_type", [torch.float32])
def test_backward(test_id, reduce, data_type):

    src_cpu = torch.tensor(tests[test_id]['src'], dtype=data_type, device='cpu')
    src_cuda = torch.tensor(tests[test_id]['src'], dtype=data_type, device='cuda')
    index_cpu = torch.tensor(tests[test_id]['index'], dtype=torch.long, device='cpu')
    index_cuda = torch.tensor(tests[test_id]['index'], dtype=torch.long, device='cuda')
    dim = tests[test_id]['dim']

    src_cpu.requires_grad_()
    src_cuda.requires_grad_()
    

    assert gradcheck(xav_dsal.scatter,
                    (src_cpu, index_cpu, dim, None, None, reduce, None, None),
                    eps=1e-4, atol=1e-2, rtol=1e-2)
    assert gradcheck(xav_dsal.scatter,
                    (src_cuda, index_cuda, dim, None, None, reduce, None, None),
                    eps=1e-4, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main(["-v", "-s", "test_scatter_autograd.py"])