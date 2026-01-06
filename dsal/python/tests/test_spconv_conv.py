import pytest
import torch
import os
import numpy as np
import logging
import importlib.util

import xav_dsal as sparse_conv_ext

def indice_conv_backward(
    features, filters, out_bp, indice_pairs, indice_pair_num, inverse=False, subm=False
):
    """
    Backward pass for indice-based convolution operation.
    
    Args:
        features: Input feature tensor
        filters: Convolution filters
        out_bp: Output backpropagation gradient
        indice_pairs: Indices for sparse convolution
        indice_pair_num: Number of index pairs
        inverse: Whether to perform inverse convolution
        subm: Whether to perform submanifold convolution
    
    Returns:
        Backward gradients
    """
    if filters.dtype == torch.float32:
        return sparse_conv_ext.indice_conv_backward_fp32(
            features, filters, out_bp, indice_pairs, indice_pair_num, int(inverse), int(subm)
        )
    elif filters.dtype == torch.half:
        return sparse_conv_ext.indice_conv_backward_half(
            features, filters, out_bp, indice_pairs, indice_pair_num, int(inverse), int(subm)
        )
    else:
        raise NotImplementedError

def indice_conv(
    features, filters, indice_pairs, indice_pair_num, num_activate_out, inverse=False, subm=False
):
    """
    Forward pass for indice-based convolution operation.
    
    Args:
        features: Input feature tensor
        filters: Convolution filters
        indice_pairs: Indices for sparse convolution
        indice_pair_num: Number of index pairs
        num_activate_out: Number of output activations
        inverse: Whether to perform inverse convolution
        subm: Whether to perform submanifold convolution
    
    Returns:
        Convolution result
    """
    if filters.dtype == torch.float32:
        return sparse_conv_ext.indice_conv_fp32(
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
            int(inverse),
            int(subm),
        )
    elif filters.dtype == torch.half:
        return sparse_conv_ext.indice_conv_half(
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
            int(inverse),
            int(subm),
        )
    else:
        raise NotImplementedError
    
def compare_tensor_precision(computed, expected, name="tensor"):
    """
    Compare two tensors and print detailed precision statistics.
    
    Args:
        computed: Tensor returned by the function
        expected: Expected tensor loaded from file
        name: Tensor name for printing
        
    Returns:
        A boolean indicating whether the tensors match within a small tolerance
    """
    if computed is None and expected is None:
        print(f"{name}: Both tensors are None. Match: Yes")
        return True
        
    if computed is None or expected is None:
        print(f"{name}: One tensor is None while the other is not. Match: No")
        return False
    
    # Check if shapes match
    if computed.shape != expected.shape:
        print(f"{name}: Shapes don't match - Computed: {computed.shape}, Expected: {expected.shape}")
        return False

    tolerance = 1e-3
    # Calculate absolute difference
    abs_sum = torch.sum(computed).item() - torch.sum(expected).item()
    print(f"{name}: Sum of absolute differences: {abs_sum}")
    abs_diff = (computed - expected).abs()
    
    # Calculate statistics
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    
    # Count elements with differences above threshold
    significant_diff_count = (abs_diff > tolerance).sum().item()
    
    # Print comparison results
    print(f"{name} comparison results:")
    print(f"  Shape: Computed {computed.shape}, Expected {expected.shape}")
    print(f"  Data type: Computed {computed.dtype}, Expected {expected.dtype}")
    print(f"  Maximum absolute difference: {max_diff}")
    print(f"  Mean absolute difference: {mean_diff}")
    print(f"  Elements with diff > {tolerance}: {significant_diff_count}/{computed.numel()} ({significant_diff_count/computed.numel()*100:.4f}%)")
    
    if significant_diff_count > 0:
        # Find indices of maximum differences
        flat_indices = abs_diff.flatten().argsort(descending=True)[:5]  # Top 5 differences
        
        # Convert flat indices to multi-dimensional indices
        if len(computed.shape) > 1:
            indices = []
            for idx in flat_indices:
                # Convert idx to multi-dimensional index
                idx_item = idx.item()
                multi_idx = []
                remaining_idx = idx_item
                for dim_size in reversed(computed.shape[1:]):
                    multi_idx.insert(0, remaining_idx % dim_size)
                    remaining_idx //= dim_size
                multi_idx.insert(0, remaining_idx)
                indices.append(tuple(multi_idx))
        else:
            indices = [(idx.item(),) for idx in flat_indices]
        
        print(f"  Top 5 differences (position, computed value, expected value, absolute difference):")
        for idx in indices:
            c_val = computed[idx].item()
            e_val = expected[idx].item()
            d_val = abs(c_val - e_val)
            print(f"    {idx}: {c_val} vs {e_val} (diff: {d_val})")
    
    return significant_diff_count == 0

def compare_indice_pairs_results(computed_out_features, out_features):
    """
    Compare out_features results with expected values.
    """
    print("=" * 50)
    print("Comparison Results")
    print("=" * 50)
    
    print("\n--- out_features comparison ---")
    out_features_match = compare_tensor_precision(computed_out_features, out_features, "out_features")
    
    # Overall assessment
    print("\n\n" + "="*70)
    print("Overall Assessment")
    print("=" * 50)
    if out_features_match:
        print("✅ All tensors match within tolerance!")
    else:
        print("❌ Differences detected in one or more tensors.")
    
    return out_features_match

def print_tensor_values(tensor1, tensor2):
    """
    Print and compare the first few values of two tensors
    
    Args:
        tensor1: First tensor to compare
        tensor2: Second tensor to compare
    """
    tensor1_flat = tensor1.flatten()
    tensor2_flat = tensor2.flatten()
    for i in range(10):
        print(f"{i}: {tensor1_flat[i].item():.8f} vs {tensor2_flat[i].item():.8f}")

def print_tensor_head(tensor, n=10):
    """
    Print the first n elements of a tensor
    
    Args:
        tensor: PyTorch tensor to print
        n: Number of elements to print, default is 10
    """
    # Save original tensor shape
    original_shape = tensor.shape
    
    # Flatten the tensor
    flattened = tensor.flatten()
    
    # Determine actual number of elements to print
    n_elements = min(n, flattened.numel())
    
    print(f"Tensor shape: {original_shape}, dtype: {tensor.dtype}")
    print(f"First {n_elements} of {flattened.numel()} elements:")
    
    # Print first n elements
    if tensor.numel() > 0:
        print(flattened[:n_elements])
    else:
        print("(Empty tensor)")
    
    # Indicate if more elements are not displayed
    if flattened.numel() > n:
        print(f"... ({flattened.numel() - n} more elements)")

# Define base path for test data as a fixture
@pytest.fixture
def base_path():
    return './data/test_spconv_conv/'

# Parametrize the test to run on multiple data groups
@pytest.mark.parametrize("group_idx", range(1, 22))  # Can be expanded to more groups as needed
def test_sparse_conv(base_path, group_idx):
    """
    Test sparse convolution with data from a specific group.
    
    Args:
        base_path: Base path for test data
        group_idx: Index of the test data group
    """
    # Build file paths
    features_file = f'{base_path}features_{group_idx}.pt'
    filters_file = f'{base_path}filters_{group_idx}.pt'
    indice_pairs_file = f'{base_path}indice_pairs_{group_idx}.pt'
    indice_pairs_num_file = f'{base_path}indice_pair_num_{group_idx}.pt'
    outids_file = f'{base_path}num_activate_out_{group_idx}.pt'
    inverse_file = f'{base_path}inverse_{group_idx}.pt'
    subm_file = f'{base_path}subm_{group_idx}.pt'
    out_features_file = f'{base_path}out_features_{group_idx}.pt'
    
    # Check if all files exist
    required_files = [
        features_file,
        filters_file,
        indice_pairs_file,
        indice_pairs_num_file,
        outids_file,
        inverse_file,
        subm_file,
        out_features_file
    ]
    for file_path in required_files:
        assert os.path.exists(file_path), f"Test file does not exist: {file_path}"
    
    # Load input data
    features = torch.load(features_file, weights_only=True)
    filters = torch.load(filters_file, weights_only=True)
    indice_pairs = torch.load(indice_pairs_file, weights_only=True)
    indice_pair_num = torch.load(indice_pairs_num_file, weights_only=True)
    outids = torch.load(outids_file, weights_only=True)
    inverse = torch.load(inverse_file, weights_only=True) 
    subm = torch.load(subm_file, weights_only=True)
    out_features = torch.load(out_features_file, weights_only=True)
    out_features = out_features.to("cpu")
    # Convert features to float32
    # features = features.to(torch.float32)
    # filters = filters.to(torch.float32)

    # Print input parameters for debugging
    print(f"\nTest group: {group_idx}")
    print("Input parameters:")
    print(f"  features shape: {features.shape}, type: {features.dtype}")
    print(f"  filters shape: {filters.shape}, type: {filters.dtype}")
    print(f"  indice_pairs shape: {indice_pairs.shape}, type: {indice_pairs.dtype}")
    print(f"  indice_pair_num shape: {indice_pair_num.shape}, type: {indice_pair_num.dtype}")
    print(f"  outids: {outids}")
    print(f"  subm: {subm}")
    
    features_cuda = features.to("cuda")
    filters_cuda = filters.to("cuda")
    indice_pairs_cuda = indice_pairs.to("cuda")
    indice_pair_num_cuda = indice_pair_num.to("cuda")    
    with torch.no_grad():
        computed_out_features = indice_conv(
            features_cuda, filters_cuda, indice_pairs_cuda, indice_pair_num_cuda, outids, inverse, subm
        )
    computed_out_features = computed_out_features.to("cpu")

    # features_cpu = features.to("cpu")
    # filters_cpu = filters.to("cpu")
    # indice_pairs_cpu = indice_pairs.to("cpu")
    # indice_pair_num_cpu = indice_pair_num.to("cpu")
    # with torch.no_grad():
    #     computed_out_features_cpu = indice_conv(
    #         features_cpu, filters_cpu, indice_pairs_cpu, indice_pair_num_cpu, outids, inverse, subm
    #     )

    # Compare values for debugging
    print("\n Values comparison(Firse 10 elements):")
    print_tensor_values(computed_out_features, out_features)
    print_tensor_head(computed_out_features, 10)
    print_tensor_head(out_features, 10)
    
    # Verify that results match expected values
    assert compare_indice_pairs_results(computed_out_features, out_features), \
        "Computed output features do not match expected values"

if __name__ == "__main__":
    pytest.main(["-v", "-s", "test_spconv_conv.py"])