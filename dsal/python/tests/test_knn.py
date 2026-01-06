import pytest
import torch
import os
import pandas as pd
import numpy as np

import xav_dsal

def check_tensor(output_cpu, output_cuda):
    if not torch.equal(output_cpu, output_cuda):
        print("Tensors are not equal!")
        print(output_cpu.shape)
        
        # Find where the tensors differ
        diff_mask = output_cpu != output_cuda
        
        # Get the indices where they differ
        diff_indices = torch.nonzero(diff_mask)
        
        # Get the differing values
        cpu_diff_values = output_cpu[diff_mask]
        cuda_diff_values = output_cuda[diff_mask]
        
        print(f"Number of differing elements: {len(diff_indices)}")
        
        # Print first N differences (to avoid flooding output)
        max_diffs_to_show = 10
        for i, idx in enumerate(diff_indices[:max_diffs_to_show]):
            idx_tuple = tuple(idx.tolist())
            print(f"Difference {i+1}:")
            print(f"  Index: {idx_tuple}")
            print(f"  CPU value: {output_cpu[idx_tuple]}")
            print(f"  CUDA value: {output_cuda[idx_tuple]}")
            print(f"  Difference: {output_cpu[idx_tuple] - output_cuda[idx_tuple]}")
        
        if len(diff_indices) > max_diffs_to_show:
            print(f"... and {len(diff_indices) - max_diffs_to_show} more differences")
        return False
    else:
        print("Tensors are equal!")
        return True
    # assert torch.equal(output_cpu, output_cuda)
    

def test_knn():
    print("\n Enter test. \n")
    # k_size = [1,16,99]
    # n_size = [500,10000]
    # m_size = [2000,10000]
    # b_size = [2,200]

    k_size = [1]
    n_size = [5000,12000]
    # m_size = [5]
    b_size = [1,50]

    same_tensor = True
    for k in k_size:
        for n in n_size:
            # for m in m_size:
                for b in b_size:
                    xyz = torch.rand(b, n, 3, dtype=torch.float32) 
                    # center_xyz = torch.rand(b, n, 3, dtype=torch.float32) 
                    # output_cpu = xav_dsal.knn(k, xyz.to("cpu"), center_xyz.to("cpu"))
                    # output_cuda = xav_dsal.knn(k, xyz.to("cuda"), center_xyz.to("cuda"))
                    output_cpu, _ = xav_dsal.knn(k, xyz.to("cpu"), None)
                    output_cuda, _ = xav_dsal.knn(k, xyz.to("cuda"), None)
                    output_cuda = output_cuda.to("cpu")

                    # xyz = torch.rand(b, 3, n, dtype=torch.float32)
                    # center_xyz = torch.rand(b, 3, m, dtype=torch.float32)
                    # output_cpu = xav_dsal.knn(k, xyz.to("cpu"), center_xyz.to("cpu"),True)
                    # output_cuda = xav_dsal.knn(k, xyz.to("cuda"), center_xyz.to("cuda"),True)
                    # output_cuda = output_cuda.to("cpu")

                    # print(output_cuda[0,:,:])

                    same_tensor = same_tensor and check_tensor(output_cpu, output_cuda)
    assert same_tensor
    print("Exit test.")
    
if __name__ == "__main__":
    pytest.main(["-v", "-s", "test_knn.py"])
