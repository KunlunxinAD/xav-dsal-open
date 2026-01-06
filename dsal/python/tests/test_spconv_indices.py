import os, sys
import torch
import pytest
import random

import xav_dsal as sparse_conv_ext

def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        size = (input_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[
            i
        ] + 1
        if kernel_size[i] == -1:
            output_size.append(1)
        else:
            output_size.append(size)
    return output_size

def get_deconv_output_size(input_size, kernel_size, stride, padding, dilation, output_padding):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        if kernel_size[i] == -1:
            raise ValueError("deconv don't support kernel_size < 0")
        size = (input_size[i] - 1) * stride[i] - 2 * padding[i] + kernel_size[i] + output_padding[i]
        output_size.append(size)
    return output_size

def get_indice_pairs(
    indices,
    batch_size,
    spatial_shape,
    ksize=3,
    stride=1,
    padding=0,
    dilation=1,
    out_padding=0,
    subm=False,
    transpose=False,
    grid=None,
):
    ndim = indices.shape[1] - 1
    if not isinstance(ksize, (list, tuple)):
        ksize = [ksize] * ndim
    if not isinstance(stride, (list, tuple)):
        stride = [stride] * ndim
    if not isinstance(padding, (list, tuple)):
        padding = [padding] * ndim
    if not isinstance(dilation, (list, tuple)):
        dilation = [dilation] * ndim
    if not isinstance(out_padding, (list, tuple)):
        out_padding = [out_padding] * ndim

    for d, s in zip(dilation, stride):
        assert any([s == 1, d == 1]), "don't support this."

    if not subm:
        if transpose:
            out_shape = get_deconv_output_size(
                spatial_shape, ksize, stride, padding, dilation, out_padding
            )
        else:
            out_shape = get_conv_output_size(spatial_shape, ksize, stride, padding, dilation)

    else:
        out_shape = spatial_shape
    if grid is None:
        if ndim == 2:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_2d
        elif ndim == 3:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_3d
        elif ndim == 4:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_4d
        else:
            raise NotImplementedError

        return get_indice_pairs_func(
            indices,
            batch_size,
            out_shape,
            spatial_shape,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            int(subm),
            int(transpose),
        )
    else:
        if ndim == 2:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_grid_2d
        elif ndim == 3:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_grid_3d
        else:
            raise NotImplementedError
        return get_indice_pairs_func(
            indices,
            grid,
            batch_size,
            out_shape,
            spatial_shape,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            int(subm),
            int(transpose),
        )

def compare_tensor_precision(computed, expected, name="tensor"):
    """
    Compare two tensors and print detailed precision statistics.
    
    Args:
        computed: Tensor returned by the function
        expected: Expected tensor loaded from file
        name: Name of the tensor for display
        
    Returns:
        Boolean indicating whether tensors match within tolerance
    """
    if computed is None and expected is None:
        print(f"{name}: Both tensors are None. Match: Yes")
        return True
        
    if computed is None or expected is None:
        print(f"{name}: One tensor is None while the other isn't. Match: No")
        return False
    
    # Verify shape match
    if computed.shape != expected.shape:
        print(f"{name}: Shape mismatch - Computed: {computed.shape}, Expected: {expected.shape}")
        return False

    tolerance = 0
    # Calculate absolute differences
    abs_diff = (computed - expected).abs()
    
    # Statistical analysis
    max_diff = abs_diff.max().item()
    significant_diff_count = (abs_diff > tolerance).sum().item()
    
    # Comparison report
    print(f"\n{name} comparison results:")
    print(f"  Shape: Computed {computed.shape}, Expected {expected.shape}")
    print(f"  Data type: Computed {computed.dtype}, Expected {expected.dtype}")
    print(f"  Maximum absolute difference: {max_diff}")
    print(f"  Elements exceeding tolerance {tolerance}: {significant_diff_count}/{computed.numel()} ({significant_diff_count/computed.numel()*100:.2f}%)")
    
    if significant_diff_count > 0:
        # Sample up to 100 discrepancies
        nonzero_indices = torch.nonzero(abs_diff, as_tuple=False)
        sample_size = min(100, len(nonzero_indices))
        sampled = nonzero_indices[:sample_size]
        
        print("\n  First 100 discrepancies (index, computed, expected, difference):")
        for idx in sampled:
            idx_tuple = tuple(idx.tolist())
            c_val = computed[idx_tuple].item()
            e_val = expected[idx_tuple].item()
            diff = abs(c_val - e_val)
            print(f"    {idx_tuple}: {c_val} vs {e_val} (Δ={diff})")
    
    return significant_diff_count == 0

def check_indice_pairs(computed_pairs, computed_outids, expected_pairs, expected_outids, indice_num):
    K, _, _ = computed_pairs.shape
    
    # Check each k
    for k in range(K):
        num_act = indice_num[k]

        # Create a dictionary to map x to a list of tuples (y, outid) for computed pairs
        computed_map = {}
        for n in range(num_act):
            x = computed_pairs[k, 0, n].item()
            y = computed_pairs[k, 1, n].item()
            if x not in computed_map:
                computed_map[x] = []
            computed_map[x].append((y, list(computed_outids[y])))
        
        # Create a dictionary to map x to a list of tuples (y, outid) for expected pairs
        expected_map = {}
        for n in range(num_act):
            x = expected_pairs[k, 0, n].item()
            y = expected_pairs[k, 1, n].item()
            if x not in expected_map:
                expected_map[x] = []
            expected_map[x].append((y, list(expected_outids[y])))
        
        # Compare the output IDs for each x
        for x in computed_map.keys():
            if x in expected_map.keys():
                # Sort the lists of tuples to ensure they match, as order of y may not matter
                computed_ys_sorted = sorted(computed_map[x], key=lambda pair: pair[0])
                expected_ys_sorted = sorted(expected_map[x], key=lambda pair: pair[0])
                
                # Check if lengths match
                if len(computed_ys_sorted) != len(expected_ys_sorted):
                    print(f"Mismatch in number of ys at k={k}, x={x}: computed {len(computed_ys_sorted)}, expected {len(expected_ys_sorted)}")
                    return False
                
                # Check if sorted y lists with outIds match
                for (computed_y, computed_outid), (expected_y, expected_outid) in zip(computed_ys_sorted, expected_ys_sorted):
                    if computed_outid != expected_outid:
                        print(f"Mismatch found at k={k}, x={x}, y={computed_y}: computed outId={computed_outid}, expected outId={expected_outid}")
                        return False
            else:
                print(f"x={x} not found in expected_map at k={k}")
                return False
    
    print("All indice pairs and output IDs match.")
    return True

def check_outids_match(computed_outids, expected_outids):
    """
    Check if two outids tensors match in content regardless of order.
    outids has shape (N, D) where each row is treated as an element.
    """
    if computed_outids.shape != expected_outids.shape:
        print(f"The dimensions don't match: computed={computed_outids.shape}, expected={expected_outids.shape}")
        return False
    print(f"the shape of two tensors are both {computed_outids.shape}.")
    
    # Convert tensors to sets of tuples (each row becomes a tuple)
    computed_rows = set(tuple(row.tolist()) for row in computed_outids)
    expected_rows = set(tuple(row.tolist()) for row in expected_outids)
    
    return computed_rows == expected_rows


def compare_results(computed_indice_pairs, computed_indice_pair_num, computed_outids,
                              expected_indice_pairs, expected_indice_pair_num, expected_outids):
    print("\n--- compare Indice Pair Num ---")
    indice_pair_num_match = compare_tensor_precision(computed_indice_pair_num, expected_indice_pair_num, "indice_pair_num")
    
    print("\n--- compare Indice Pairs ---")
    indice_pairs_match = check_indice_pairs(computed_indice_pairs, computed_outids,
                                               expected_indice_pairs, expected_outids, expected_indice_pair_num)
    
    print("\n--- compare Outids ---")
    outids_match = check_outids_match(computed_outids, expected_outids)
    
    print("\n\n" + "="*70)
    if indice_pairs_match and indice_pair_num_match and outids_match:
        print("all matches")
    else:
        print("failed to pass:")
        if not indice_pairs_match:
            print("  - indice_pairs not matched")
        if not indice_pair_num_match:
            print("  - indice_pair_num not matched")
        if not outids_match:
            print("  - outids not matched")
    
    return indice_pairs_match and indice_pair_num_match and outids_match

from pathlib import Path
SCRIPT_DIR = Path(__file__).parent.resolve()
DEBUG_DIR = SCRIPT_DIR / "data/test_spconv_indice"

def get_debug_files(prefix, sample_size=10):
    if not DEBUG_DIR.exists():
        raise FileNotFoundError(f"Directory '{DEBUG_DIR}' does not exist.")
    files = sorted(DEBUG_DIR.glob(f"{prefix}*.pt"))
    if sample_size:
        return random.sample(files, min(sample_size, len(files)))
    else:
        return files4

@pytest.mark.parametrize("fwd_file", get_debug_files("indices"))
# @pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("device", ["cuda"])
def test_get_indice_subm(fwd_file, device):
    group_idx = int(fwd_file.stem.split('_')[-1])  # 提取数字编号，比如 indices_3.pt -> 3
    base_path = str(fwd_file.parent) + "/"
    
    print("\n" + "="*70)
    print(f"Testing group {group_idx} on {device}")
    print("="*70)

    # Build file paths
    files = {
        'indices': f'{base_path}indices_{group_idx}.pt',
        'batch_size': f'{base_path}batch_size_{group_idx}.pt',
        'spatial_shape': f'{base_path}spatial_shape_{group_idx}.pt',
        'ksize': f'{base_path}kernel_size_{group_idx}.pt',
        'stride': f'{base_path}stride_{group_idx}.pt',
        'padding': f'{base_path}padding_{group_idx}.pt',
        'dilation': f'{base_path}dilation_{group_idx}.pt',
        'out_padding': f'{base_path}output_padding_{group_idx}.pt',
        'subm': f'{base_path}subm_{group_idx}.pt',
        'transpose': f'{base_path}transposed_{group_idx}.pt',
        'indice_pairs': f'{base_path}indice_pairs{group_idx}.pt',
        'indice_pair_num': f'{base_path}indice_pair_num{group_idx}.pt',
        'outids': f'{base_path}outids{group_idx}.pt'
    }

    # Check all files exist
    if not all(os.path.exists(f) for f in files.values()):
        assert False, f"Missing files in group {group_idx}"

    # Load and convert parameters
    indices = torch.load(files['indices'], weights_only=True).to(device)
    batch_size = torch.load(files['batch_size'], weights_only=True)
    spatial_shape = list(torch.load(files['spatial_shape'], weights_only=True))
    ksize = list(torch.load(files['ksize'], weights_only=True))
    stride = list(torch.load(files['stride'], weights_only=True))
    padding = list(torch.load(files['padding'], weights_only=True))
    dilation = list(torch.load(files['dilation'], weights_only=True))
    out_padding = list(torch.load(files['out_padding'], weights_only=True))
    subm = torch.load(files['subm'], weights_only=True)
    transpose = torch.load(files['transpose'], weights_only=True)

    # Load expected outputs (CPU tensors)
    expected = {
        'indice_pairs': torch.load(files['indice_pairs'], weights_only=True),
        'indice_pair_num': torch.load(files['indice_pair_num'], weights_only=True),
        'out_indices': torch.load(files['outids'], weights_only=True)
    }

    print("Input parameters:")
    print(f"  indices shape: {indices.shape}, dtype: {indices.dtype}")
    print(f"  batch_size: {batch_size}")
    print(f"  spatial_shape: {spatial_shape}")
    print(f"  kernel_size: {ksize}")
    print(f"  stride: {stride}")
    print(f"  padding: {padding}")
    print(f"  dilation: {dilation}")
    print(f"  output_padding: {out_padding}")
    print(f"  subm: {subm}")
    print(f"  transpose: {transpose}\n")

    # Compute results on specified device
    computed = get_indice_pairs(
        indices, batch_size, spatial_shape, ksize, stride,
        padding, dilation, out_padding, subm, transpose
    )
    
    # Move results to CPU for comparison
    computed_cpu = (
        computed[0].cpu(),
        computed[1].cpu(),
        computed[2].cpu()
    )

    # Compare results
    match = compare_results(
        computed_cpu[1], computed_cpu[2], computed_cpu[0],
        expected['indice_pairs'], expected['indice_pair_num'], expected['out_indices']
    )
    assert match, f"Group {group_idx} failed on {device}"           

if __name__ == "__main__":
    group_idx = None

    if len(sys.argv) >= 2:
        input_str = sys.argv[1]
        try:
            group_idx = int(input_str)
            print(f"test group: {group_idx}")
        except ValueError:
            print(f"parameter should be int instead of '{input_str}'")
            sys.exit(1)  

    if group_idx is not None:
        fwd_file = DEBUG_DIR / f"indices_{group_idx}.pt"
        if not fwd_file.exists():
            print(f"File {fwd_file} does not exist!")
            sys.exit(1)
        test_get_indice_subm(fwd_file, "cuda")
        test_get_indice_subm(fwd_file, "cpu")
    else:
        for f in get_debug_files("indices", None):
            test_get_indice_subm(f, "cuda")
            test_get_indice_subm(f, "cpu")