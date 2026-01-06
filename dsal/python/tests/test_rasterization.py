import sys
import torch
import random
from torch.autograd import Function
from pathlib import Path
import pytest
import xav_dsal

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

        contribution_map = torch.IntTensor(
            ctx.batch_size,
            ctx.height,
            ctx.width).fill_(0).to(device=ctx.device)
        rasterized, contribution_map = xav_dsal.forward_rasterize(vertices, rasterized, contribution_map, width, height, inv_smoothness, ctx.mode)
        ctx.save_for_backward(vertices, rasterized, contribution_map)

        return rasterized #, contribution_map

    @staticmethod
    def backward(ctx, grad_output):
        vertices, rasterized, contribution_map = ctx.saved_tensors

        grad_output = grad_output.contiguous()

        grad_vertices = torch.FloatTensor(ctx.batch_size, ctx.number_vertices, 2).fill_(0.0).to(device=ctx.device)

        grad_vertices = xav_dsal.backward_rasterize(
            vertices, rasterized, contribution_map, grad_output, grad_vertices, ctx.width, ctx.height, ctx.inv_smoothness, ctx.mode)

        return grad_vertices, None, None, None, None
    
SCRIPT_DIR = Path(__file__).parent.resolve()
DEBUG_DIR = SCRIPT_DIR / "data/test_rasterization" 
def get_debug_files(prefix, sample_size=10):
    if not DEBUG_DIR.exists():
        raise FileNotFoundError(f"Directory '{DEBUG_DIR}' does not exist.")
    files = sorted(DEBUG_DIR.glob(f"{prefix}*.pth"))
    if sample_size:
        return random.sample(files, min(sample_size, len(files)))
    else:
        return files

import torch
import pytest


# 假设 get_debug_files 已经定义好了
# from your_module import get_debug_files, SoftPolygonFunction

@pytest.mark.parametrize("fwd_file", get_debug_files("fwd"))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward_pass(fwd_file, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    data = torch.load(fwd_file, weights_only=True)
    fwd_in = data['in']
    expected_out = data['out']['rasterized']

    vertices = fwd_in['vertices'].to(device).requires_grad_()
    width = fwd_in['width']
    height = fwd_in['height']
    inv_smoothness = fwd_in['inv_smoothness']
    mode = fwd_in['mode']

    output = SoftPolygonFunction.apply(vertices, width, height, inv_smoothness, mode)

    # 将 expected_out 移动到相同设备上再比较
    expected_out_device = expected_out.to(device)

    assert output.shape == expected_out.shape, f"Shape mismatch: {output.shape} vs {expected_out.shape}"

    max_diff = (output - expected_out_device).abs().max().item()
    print(f"[{device}] Forward pass diff: {max_diff}, vertices shape: {vertices.shape}, width: {width}, height: {height}, smoothness: {inv_smoothness:.3f}, mode: {mode}")
    assert max_diff < 1e-4


@pytest.mark.parametrize("bkwd_file", get_debug_files("bkwd"))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_backward_pass(bkwd_file, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    data = torch.load(bkwd_file, weights_only=True)
    grad_output = data['grad_output'].to(device)
    saved_tensors = data['saved_tensors']

    # Move saved tensors to target device
    saved_tensors_device = tuple(t.to(device) for t in saved_tensors)
    vertices, rasterized, contribution_map = saved_tensors_device

    ctx = type('', (), {})()
    ctx.saved_tensors = saved_tensors_device
    ctx.width = data['width']
    ctx.height = data['height']
    ctx.inv_smoothness = data['inv_smoothness']
    ctx.mode = data['mode']
    ctx.batch_size = vertices.shape[0]
    ctx.number_vertices = vertices.shape[1]
    ctx.device = device

    expected_grad_vertices = data['grad_vertices'].to(device)

    grad_vertices, *_ = SoftPolygonFunction.backward(ctx, grad_output)

    assert isinstance(grad_vertices, torch.Tensor), "Backward must return a tensor for vertices"
    assert grad_vertices.shape == expected_grad_vertices.shape, f"Grad shape mismatch: {grad_vertices.shape} vs {expected_grad_vertices.shape}"

    max_diff = (grad_vertices - expected_grad_vertices).abs().max().item()
    print(f"[{device}] Backward pass diff: {max_diff}")
    assert max_diff < 1e-4

if __name__ == "__main__":
    device = "cuda"
    device = "cpu"
    idx = None

    if len(sys.argv) >= 2:
        input_str = sys.argv[1]
        try:
            idx = int(input_str)
            print(f"test file: {idx}")
        except ValueError:
            print(f"parameter should be int instead of '{input_str}'")
            sys.exit(1)  
    if idx:
        fwd_file = get_debug_files(f"fwd{idx}")[0]
        bkwd_file = get_debug_files(f"bkwd{idx}")[0]
        test_forward_pass(fwd_file, device)
        test_backward_pass(bkwd_file, device)
    else:
        # test all cases
        for f in get_debug_files("fwd", None):
            print("test",f)
            test_forward_pass(f, device)
        for f in get_debug_files("bkwd", None):
            print("test",f)
            test_backward_pass(f, device)

# def pnp(vertices, width, height):
#     device = vertices.device
#     batch_size = vertices.size(0)
#     polygon_dimension = vertices.size(1)

#     y_index = torch.arange(0, height).to(device)
#     x_index = torch.arange(0, width).to(device)
    
#     grid_y, grid_x = torch.meshgrid(y_index, x_index)
#     xp = grid_x.unsqueeze(0).repeat(batch_size, 1, 1).float()
#     yp = grid_y.unsqueeze(0).repeat(batch_size, 1, 1).float()

#     result = torch.zeros((batch_size, height, width)).bool().to(device)

#     j = polygon_dimension - 1
#     for vn in range(polygon_dimension):
#         from_x = vertices[:, vn, 0].unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)        
#         from_y = vertices[:, vn, 1].unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)

#         to_x = vertices[:, j, 0].unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)
#         to_y = vertices[:, j, 1].unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)

#         has_condition = torch.logical_and((from_y > yp) != (to_y > yp), xp < ((to_x - from_x) * (yp - from_y) / (to_y - from_y) + from_x))
        
#         if has_condition.any():
#             result[has_condition] = ~result[has_condition]

#         j = vn

#     signed_result = -torch.ones((batch_size, height, width), device=device)
#     signed_result[result] = 1.0

#     return signed_result


# # used for verification purposes only.
# class SoftPolygonPyTorch(nn.Module):
#     def __init__(self, inv_smoothness=1.0):
#         super(SoftPolygonPyTorch, self).__init__()

#         self.inv_smoothness = inv_smoothness

#     # vertices is N x P x 2
#     # todo, implement inside outside.
#     def forward(self, vertices, width, height, p, color=False):
#         device = vertices.device
#         batch_size = vertices.size(0)
#         polygon_dimension = vertices.size(1)

#         inside_outside = pnp(vertices, width, height)

#         # discrete points we will sample from.
#         y_index = torch.arange(0, height).to(device)
#         x_index = torch.arange(0, width).to(device)
        
#         grid_y, grid_x = torch.meshgrid(y_index, x_index)
#         grid_x = grid_x.unsqueeze(0).repeat(batch_size, 1, 1).float()
#         grid_y = grid_y.unsqueeze(0).repeat(batch_size, 1, 1).float()
        
#         # do this "per dimension"
#         distance_segments = []
#         over_segments = []
#         color_segments = []
#         for from_index in range(polygon_dimension):
#             segment_result = torch.zeros((batch_size, height, width)).to(device)
#             from_vertex = vertices[:, from_index].unsqueeze(-1).unsqueeze(-1)
            
#             if from_index == (polygon_dimension - 1):
#                 to_vertex = vertices[:, 0].unsqueeze(-1).unsqueeze(-1)
#             else:
#                 to_vertex = vertices[:, from_index + 1].unsqueeze(-1).unsqueeze(-1)
                
#             x2_sub_x1 = to_vertex[:, 0] - from_vertex[:, 0]
#             y2_sub_y1 = to_vertex[:, 1] - from_vertex[:, 1]
#             square_segment_length = x2_sub_x1 * x2_sub_x1 + y2_sub_y1 * y2_sub_y1 + 0.00001
            
#             # figure out if this is a major/minor segment (todo?)
#             x_sub_x1 = grid_x - from_vertex[:, 0]
#             y_sub_y1 = grid_y - from_vertex[:, 1]
#             x_sub_x2 = grid_x - to_vertex[:, 0]
#             y_sub_y2 = grid_y - to_vertex[:, 1]
            
#             # dot between the given point and first vertex and first vertex and second vertex.
#             dot = ((x_sub_x1 * x2_sub_x1) + (y_sub_y1 * y2_sub_y1)) / square_segment_length

#             # needlessly computed sometimes.
#             x_proj = grid_x - (from_vertex[:, 0] + dot * x2_sub_x1)
#             y_proj = grid_y - (from_vertex[:, 1] + dot * y2_sub_y1)

#             from_closest = dot < 0
#             to_closest = dot > 1
#             interior_closest = (dot >= 0) & (dot <= 1)

#             segment_result[from_closest] = x_sub_x1[from_closest] ** 2 + y_sub_y1[from_closest] ** 2
#             segment_result[to_closest] = x_sub_x2[to_closest] ** 2 + y_sub_y2[to_closest] ** 2
#             segment_result[interior_closest] = x_proj[interior_closest] ** 2 + y_proj[interior_closest] ** 2

#             distance_map = -segment_result
#             distance_segments.append(distance_map)

#             signed_map = torch.sigmoid(-distance_map * inside_outside / self.inv_smoothness)
#             over_segments.append(signed_map)

#         F_max, F_arg = torch.max(torch.stack(distance_segments, dim=-1), dim=-1)
#         F_theta = torch.gather(torch.stack(over_segments, dim=-1), dim=-1, index=F_arg.unsqueeze(-1))[..., 0]

#         return F_theta
