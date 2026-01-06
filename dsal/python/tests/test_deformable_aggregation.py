# Copyright(c) OpenMMLab.All rights reserved.
import unittest
from typing import Optional, Union
import torch
import numpy as np
import pytest
import os
from data_compare import compare_tensors

# from mmcv.utils import ext_loader
# ext_module = ext_loader.load_ext('_ext',
#         ['deformable_aggregation_forward'])

import xav_dsal as ext_module

def check_and_make_contiguous(tensor, name):
    if not tensor.is_contiguous():
        print(f"{name} is not contiguous. Converting to contiguous.")
        return tensor.contiguous()
    return tensor


class DeformableAggregationFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):
        mc_ms_feat = check_and_make_contiguous(mc_ms_feat, "mc_ms_feat")
        spatial_shape = check_and_make_contiguous(spatial_shape, "spatial_shape")
        scale_start_index = check_and_make_contiguous(scale_start_index, "scale_start_index")
        sampling_location = check_and_make_contiguous(sampling_location, "sampling_location")
        weights = check_and_make_contiguous(weights, "weights")

        output = ext_module.deformable_aggregation_forward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )

        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors

        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        
        mc_ms_feat = check_and_make_contiguous(mc_ms_feat, "mc_ms_feat")
        spacial_shape = check_and_make_contiguous(spatial_shape, "spatial_shape")
        scale_start_index = check_and_make_contiguous(scale_start_index, "scale_start_index")
        sampling_location = check_and_make_contiguous(sampling_location, "sampling_location")
        weights = check_and_make_contiguous(weights, "weights")
        grad_output = check_and_make_contiguous(grad_output, "grad_output")
        ext_module.deformable_aggregation_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output,
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )

        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
        )


torch.manual_seed(2)
class TestDeformableAggregation(unittest.TestCase):
    batch = 1
    cam = 1
    scale = 1
    anchor = 5
    pts = 1
    group = 2
    feat = 128
    c = 8
    dims = [
        # [batch, anchor, pts, cam, scale,   c,    feat, group],
        [    1,     1,  1,   6,     4,    256,   -1,     8],
        # shape test
        # [    1,     1,  1,   5,     4,    128,   -1,     2],
        [    1,     1,  1,   5,     4,     96,   -1,     3],
        [    1,     1,  1,   7,     8,     64,   -1,     2],
        [    1,     1,  1,   3,     7,     32,   -1,     1],
    ]
    mc_ms_feat = []
    spatial_shape = []
    scale_start_index = []
    sampling_locations = []
    weights = []

    def reset_parameters(self):
        self.mc_ms_feat = []
        self.spatial_shape = []
        self.scale_start_index = []
        self.sampling_locations = []
        self.weights = []

    def _test_deform_agg(self, dtype, device):
        outputs = []
        grads = {"mc_ms_feat": [], "sampling_locations": [], "weights": []}

        for i in range(len(self.mc_ms_feat)):
            mc_ms_feat = self.mc_ms_feat[i].to(device).detach()
            mc_ms_feat.requires_grad = True
            mc_ms_feat.retain_grad()

            spacial_shape = self.spatial_shape[i].to(device).detach()
            scale_start_index = self.scale_start_index[i].to(device).detach()

            sampling_location = self.sampling_locations[i].to(device).detach()
            sampling_location.requires_grad = True
            sampling_location.retain_grad()

            weights = self.weights[i].to(device).detach()
            weights.requires_grad = True
            weights.retain_grad()

            out = DeformableAggregationFunction.apply(
                mc_ms_feat,
                spacial_shape,
                scale_start_index,
                sampling_location,
                weights,
            )
            # print(device, out)
            outputs.append(out.clone())

            out.sum().backward()

            grads["mc_ms_feat"].append(mc_ms_feat.grad.clone())
            grads["sampling_locations"].append(sampling_location.grad.clone())
            grads["weights"].append(weights.grad.clone())

        return outputs, grads


    def test_deformable_aggregation(self):
        self.reset_parameters()
        dtype = torch.float32
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'data/deformable_inputs_case1.pt')
        ckpt = torch.load(file_path, weights_only=True)

        self.mc_ms_feat.append(ckpt["mc_ms_feat"])
        self.spatial_shape.append(ckpt["spatial_shape"].to(dtype=torch.int32))
        self.sampling_locations.append(ckpt["sampling_location"])
        self.weights.append(ckpt["weights"])
        self.scale_start_index.append(ckpt["scale_start_index"].to(dtype=torch.int32))
        batch, feat, c = ckpt["mc_ms_feat"].shape
        cam, scale = ckpt["spatial_shape"].shape[:2]
        anchor, pts = ckpt["sampling_location"].shape[1:3]
        group = ckpt["weights"].shape[-1]
        print(batch, feat, c, cam, scale, anchor, pts, group)

        output_gpu, grads_gpu = self._test_deform_agg(dtype, "cuda")
        output_cpu, grads_cpu = self._test_deform_agg(dtype, "cpu")

        for i in range(len(output_cpu)):
            np.testing.assert_allclose(
                output_cpu[i].detach().numpy(),
                output_gpu[i].detach().cpu().numpy(),
                atol=1e-3,
            )
        
        for key in grads_gpu:
            print(f"\nComparing gradients for key: {key}")
            for idx, (grad_cpu, grad_gpu) in enumerate(zip(grads_cpu[key], grads_gpu[key])):
                grad_cpu_array = grad_cpu.detach().cpu()
                grad_gpu_array = grad_gpu.detach().cpu()

                np.testing.assert_allclose(
                    grad_cpu_array,
                    grad_gpu_array,
                    rtol=1e-3,
                    atol=1e-3,
                    err_msg=f"Gradient mismatch for {key} at shape ({self.dims[idx]})",
                )

    def test_deformable_aggregation_random(self):
        self.reset_parameters()
        dtype = torch.float32
        use_random_data = False
        for dim in self.dims:
            print(*dim)
            batch, anchor, pts, cam, scale, embeds, feat, group = dim
            
            # set shape
            self.spatial_shape.append(torch.randint(1, 10, [cam, scale, 2], dtype=torch.int32))
                
            # compute feat
            feat = 0
            scale_start_index = []
            for c in range(cam):
                for s in range(scale):
                    feat += self.spatial_shape[-1][c][s][0] * self.spatial_shape[-1][c][s][1]
                    scale_start_index.append(int(feat))
            scale_start_index = [0] + scale_start_index[:-1]
            self.scale_start_index.append(torch.tensor(scale_start_index, dtype=torch.int32))

            if use_random_data:
                self.mc_ms_feat.append(torch.randn((batch, feat, embeds), dtype=dtype, requires_grad=True))
                self.sampling_locations.append(torch.rand((batch, anchor, pts, cam, scale, 2), dtype=dtype, requires_grad=True))
                self.weights.append(torch.randn((batch, anchor, pts, cam, scale, group), dtype=dtype, requires_grad=True))
            else:
                self.mc_ms_feat.append(torch.full((batch, feat, embeds), 1, dtype=dtype, requires_grad=True))
                self.sampling_locations.append(torch.full((batch, anchor, pts, cam, scale, 2), 0.5, dtype=dtype, requires_grad=True))
                self.weights.append(torch.full((batch, anchor, pts, cam, scale, group), 0.0038, dtype=dtype, requires_grad=True))
 

        output_gpu, grads_gpu = self._test_deform_agg(dtype, "cuda")
        output_cpu, grads_cpu = self._test_deform_agg(dtype, "cpu")

        for i in range(len(output_cpu)):
            np.testing.assert_allclose(
                output_cpu[i].detach().numpy(),
                output_gpu[i].detach().cpu().numpy(),
                atol=1e-3,
            )
        
        for key in grads_gpu:
            print(f"\nComparing gradients for key: {key}")
            for idx, (grad_cpu, grad_gpu) in enumerate(zip(grads_cpu[key], grads_gpu[key])):
                grad_cpu_array = grad_cpu.detach().cpu()
                grad_gpu_array = grad_gpu.detach().cpu()

                np.testing.assert_allclose(
                    grad_cpu_array,
                    grad_gpu_array,
                    rtol=1e-3,
                    atol=1e-3,
                    err_msg=f"Gradient mismatch for {key} at shape ({self.dims[idx]})",
                )
                
def run_fwd(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights, variant):
    print("mc_ms_feat shape", mc_ms_feat.shape, mc_ms_feat.dtype, mc_ms_feat.device)
    print("spatial_shape shape", spatial_shape.shape, spatial_shape.dtype, spatial_shape.device)
    print("scale_start_index shape", scale_start_index.shape, scale_start_index.dtype, scale_start_index.device)
    print("sampling_location shape", sampling_location.shape, sampling_location.dtype, sampling_location.device)
    print("weights shape", weights.shape, weights.dtype, weights.device)

    output = ext_module.deformable_aggregation_forward(
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
        variant
    )
    output = output.cpu()
    print("output shape", output.shape, output.dtype)
    return output

def run_bwd(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights, grad_output, variant):
    print("mc_ms_feat shape", mc_ms_feat.shape, mc_ms_feat.dtype, mc_ms_feat.device)
    print("spatial_shape shape", spatial_shape.shape, spatial_shape.dtype, spatial_shape.device)
    print("scale_start_index shape", scale_start_index.shape, scale_start_index.dtype, scale_start_index.device)
    print("sampling_location shape", sampling_location.shape, sampling_location.dtype, sampling_location.device)
    print("weights shape", weights.shape, weights.dtype, weights.device)
    print("grad_output shape", grad_output.shape, grad_output.dtype, grad_output.device)

    grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
    grad_sampling_location = torch.zeros_like(sampling_location)
    grad_weights = torch.zeros_like(weights)
    # print("grad_mc_ms_feat shape", grad_mc_ms_feat.shape, grad_mc_ms_feat.dtype, grad_mc_ms_feat.device)
    # print("grad_sampling_location shape", grad_sampling_location.shape, grad_sampling_location.dtype, grad_sampling_location.device)
    # print("grad_weights shape", grad_weights.shape, grad_weights.dtype, grad_weights.device)
    ext_module.deformable_aggregation_backward(
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
        grad_output,
        grad_mc_ms_feat,
        grad_sampling_location,
        grad_weights,
        variant
    )
    grad_mc_ms_feat = grad_mc_ms_feat.cpu()
    grad_sampling_location = grad_sampling_location.cpu()
    grad_weights = grad_weights.cpu()
    print("grad_mc_ms_feat shape", grad_mc_ms_feat.shape, grad_mc_ms_feat.dtype)
    print("grad_sampling_location shape", grad_sampling_location.shape, grad_sampling_location.dtype)
    print("grad_weights shape", grad_weights.shape, grad_weights.dtype)
    return (grad_mc_ms_feat, grad_sampling_location, grad_weights)

def test_deformable_aggregation_variant_fwd():
    file_path = "./data/test_deformable_aggregation_variant/deformable_aggregation_test_input.pt"
    data = torch.load(file_path, weights_only=True)

    #### new version 
    output_new = run_fwd(data['mc_ms_feat'], data['spatial_shape'], data['scale_start_index'], data['sampling_location'], data['weights'], True)
    torch.save(output_new, "./data/test_deformable_aggregation_variant/deformable_aggregation_test_output.pt")

    #### origin version 
    mc_ms_feat = data['mc_ms_feat']
    spatial_shape = data['spatial_shape']
    scale_start_index = data['scale_start_index']
    sampling_location = data['sampling_location']
    weights = data['weights']

    mc_ms_feat = mc_ms_feat.reshape(mc_ms_feat.shape[0], -1, mc_ms_feat.shape[3]).contiguous().float()
    spatial_shape = spatial_shape.unsqueeze(0).expand(6, spatial_shape.shape[0], spatial_shape.shape[1]).contiguous().int()
    feature_length = mc_ms_feat.shape[1]
    first_start_idx = scale_start_index
    group_len = feature_length // 6
    group_base = torch.arange(6, dtype=torch.int32) * group_len
    relative_offsets = first_start_idx - scale_start_index[0]
    scale_start_index = group_base.unsqueeze(1).to("cuda") + relative_offsets.unsqueeze(0)  
    scale_start_index = scale_start_index.contiguous().int()
    sampling_location = sampling_location.unsqueeze(2).contiguous().float()
    weights = weights.unsqueeze(2).contiguous().float()

    output_origin = run_fwd(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights, False)

    assert(compare_tensors(output_new, output_origin, 5e-4, 3e-4))

def test_deformable_aggregation_variant_bwd():
    input_file_path = "./data/test_deformable_aggregation_variant/deformable_aggregation_test_input.pt"
    output_file_path = "./data/test_deformable_aggregation_variant/deformable_aggregation_test_output.pt"
    intput_data = torch.load(input_file_path, weights_only=True)
    output_data = torch.load(output_file_path, weights_only=True).to("cuda")

    (grad_mc_ms_feat_new, grad_sampling_location_new, grad_weights_new) = run_bwd(intput_data['mc_ms_feat'], 
                intput_data['spatial_shape'], intput_data['scale_start_index'],
                intput_data['sampling_location'], intput_data['weights'], output_data, True)

    #### origin version 
    mc_ms_feat = intput_data['mc_ms_feat']
    spatial_shape = intput_data['spatial_shape']
    scale_start_index = intput_data['scale_start_index']
    sampling_location = intput_data['sampling_location']
    weights = intput_data['weights']

    mc_ms_feat_shape = mc_ms_feat.shape
    mc_ms_feat = mc_ms_feat.reshape(mc_ms_feat.shape[0], -1, mc_ms_feat.shape[3]).contiguous().float()
    spatial_shape = spatial_shape.unsqueeze(0).expand(6, spatial_shape.shape[0], spatial_shape.shape[1]).contiguous().int()
    feature_length = mc_ms_feat.shape[1]
    first_start_idx = scale_start_index
    group_len = feature_length // 6
    group_base = torch.arange(6, dtype=torch.int32) * group_len
    relative_offsets = first_start_idx - scale_start_index[0]
    scale_start_index = group_base.unsqueeze(1).to("cuda") + relative_offsets.unsqueeze(0)  
    scale_start_index = scale_start_index.contiguous().int()
    sampling_location = sampling_location.unsqueeze(2).contiguous().float()
    weights = weights.unsqueeze(2).contiguous().float()
   
    (grad_mc_ms_feat, grad_sampling_location, grad_weights) = run_bwd(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights, output_data, False)

    grad_mc_ms_feat = grad_mc_ms_feat.reshape(mc_ms_feat_shape)
    grad_sampling_location = grad_sampling_location.squeeze(2)
    grad_weights = grad_weights.squeeze(2)

    assert(compare_tensors(grad_mc_ms_feat_new, grad_mc_ms_feat, 5e-4, 3e-4))
    assert(compare_tensors(grad_sampling_location_new, grad_sampling_location, 5e-4, 3e-4))
    assert(compare_tensors(grad_weights_new, grad_weights, 5e-4, 3e-4))

if __name__ == "__main__":
    ut = TestDeformableAggregation()
    ut.test_deformable_aggregation()
    ut.test_deformable_aggregation_random()
    test_deformable_aggregation_variant_fwd()
    test_deformable_aggregation_variant_bwd()