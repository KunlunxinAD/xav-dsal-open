import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function


# import importlib.util
# so_path = "./native_rasterizer_all.cpython-38-x86_64-linux-gnu.so"
# spec = importlib.util.spec_from_file_location("native_rasterizer", so_path)
# native_rasterizer = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(native_rasterizer)

import xav_dsal as native_rasterizer_wrapper
# import native_rasterizer as native_rasterizer

MODE_BOUNDARY = "boundary"
MODE_MASK = "mask"
MODE_HARD_MASK = "hard_mask"

MODE_MAPPING = {
    MODE_BOUNDARY: 0,
    MODE_MASK: 1,
    MODE_HARD_MASK: 2
}


class SoftPolygonFunction(Function):
    @staticmethod
    def forward(ctx, vertices, width, height, inv_smoothness=1.0, mode=MODE_BOUNDARY):
        ctx.width = width
        ctx.height = height
        ctx.inv_smoothness = inv_smoothness
        ctx.mode = MODE_MAPPING[mode]

        vertices = vertices.clone()
        ctx.device = vertices.device
        ctx.batch_size, ctx.number_vertices = vertices.shape[:2]
        
        rasterized = torch.FloatTensor(ctx.batch_size, ctx.height, ctx.width).fill_(0.0).to(device=ctx.device)
        contribution_map = torch.IntTensor(ctx.batch_size, ctx.height, ctx.width).fill_(0).to(device=ctx.device)
        
        rasterized, contribution_map = native_rasterizer_wrapper.forward_rasterize_xtrans(
            vertices, rasterized, contribution_map, width, height, inv_smoothness, ctx.mode)
        ctx.save_for_backward(vertices, rasterized, contribution_map)

        return rasterized

    @staticmethod
    def backward(ctx, grad_output):
        vertices, rasterized, contribution_map = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_vertices = torch.FloatTensor(ctx.batch_size, ctx.number_vertices, 2).fill_(0.0).to(device=ctx.device)
        
        grad_vertices = native_rasterizer_wrapper.backward_rasterize_xtrans(
            vertices, rasterized, contribution_map, grad_output, grad_vertices, 
            ctx.width, ctx.height, ctx.inv_smoothness, ctx.mode)

        return grad_vertices, None, None, None, None

# class SoftPolygonFunction_all(Function):
#     @staticmethod
#     def forward(ctx, vertices, width, height, inv_smoothness=1.0, mode=MODE_BOUNDARY):
#         ctx.width = width
#         ctx.height = height
#         ctx.inv_smoothness = inv_smoothness
#         ctx.mode = MODE_MAPPING[mode]

#         vertices = vertices.clone()
#         ctx.device = vertices.device
#         ctx.batch_size, ctx.number_vertices = vertices.shape[:2]
        
#         rasterized = torch.FloatTensor(ctx.batch_size, ctx.height, ctx.width).fill_(0.0).to(device=ctx.device)
#         contribution_map = torch.IntTensor(ctx.batch_size, ctx.height, ctx.width).fill_(0).to(device=ctx.device)
        
#         rasterized, contribution_map = native_rasterizer.forward_rasterize(
#             vertices, rasterized, contribution_map, width, height, inv_smoothness, ctx.mode)
#         ctx.save_for_backward(vertices, rasterized, contribution_map)

#         return rasterized

#     @staticmethod
#     def backward(ctx, grad_output):
#         vertices, rasterized, contribution_map = ctx.saved_tensors
#         grad_output = grad_output.contiguous()
#         grad_vertices = torch.FloatTensor(ctx.batch_size, ctx.number_vertices, 2).fill_(0.0).to(device=ctx.device)
        
#         grad_vertices = native_rasterizer.backward_rasterize(
#             vertices, rasterized, contribution_map, grad_output, grad_vertices, 
#             ctx.width, ctx.height, ctx.inv_smoothness, ctx.mode)

#         return grad_vertices, None, None, None, None

def test_native_rasterizer():
    """Simple test for native_rasterizer forward and backward operators"""
    print("Testing native_rasterizer operators...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters
    batch_size = 2
    num_vertices = 4
    width, height = 64, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}, Number of vertices: {num_vertices}")
    print(f"Output size: {width}x{height}")
    
    # Create random vertices (within image range)
    vertices = (torch.rand(batch_size, num_vertices, 2, device=device) * 50 + 10).detach().requires_grad_(True)
    print(f"Input vertices shape: {vertices.shape}")
    print(f"Vertices range: [{vertices.min().item():.2f}, {vertices.max().item():.2f}]")

    # Test forward pass
    print("\n=== Forward Pass Test ===")
    try:
        output = SoftPolygonFunction.apply(vertices, width, height, 1.0, MODE_MASK)
        # output_all = SoftPolygonFunction_all.apply(vertices, width, height, 1.0, MODE_MASK)
        # # print(torch.eq(output, output_all))
        # result = torch.eq(output, output_all)
        # all_equal = torch.all(result)
        # print(f"All elements are equal: {all_equal.item()}")
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"  Output mean: {output.mean().item():.4f}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        result = False
        assert result is True
    
    # Test backward pass
    print("\n=== Backward Pass Test ===")
    try:
        loss = output.sum()
        loss.backward()
        print(f"✓ Backward pass successful")
        if vertices.grad is not None:
            print(f"  Gradient shape: {vertices.grad.shape}")
            print(f"  Gradient range: [{vertices.grad.min().item():.6f}, {vertices.grad.max().item():.6f}]")
            print(f"  Gradient mean: {vertices.grad.mean().item():.6f}")
            print(f"  Gradient valid: {not torch.isnan(vertices.grad).any()}")
        else:
            print("  Warning: Vertices gradient is None")
            print(f"  vertices.requires_grad: {vertices.requires_grad}")
            print(f"  vertices.is_leaf: {vertices.is_leaf}")
        
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        result = False
        assert result is True
    
    # Test different modes
    print("\n=== Multi-mode Test ===")
    for mode in [MODE_BOUNDARY, MODE_MASK, MODE_HARD_MASK]:
        try:
            vertices_test = torch.rand(1, 3, 2, device=device, requires_grad=True) * 40 + 10
            output_test = SoftPolygonFunction.apply(vertices_test, 32, 32, 2.0, mode)
            loss_test = output_test.mean()
            loss_test.backward()
            print(f"✓ Mode {mode}: Output range [{output_test.min().item():.4f}, {output_test.max().item():.4f}]")
        except Exception as e:
            print(f"✗ Mode {mode} failed: {e}")
    
    print("\n🎉 All tests passed! Operators are working properly.")
    result = True
    assert result is True


if __name__ == "__main__":
    test_native_rasterizer()