import os
import math
import unittest
from typing import List, Optional, Tuple, Union

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single

import xav_dsal

class DCNv4Function(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, input, offset_mask,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, dilation_h, dilation_w,
            group, group_channels, offset_scale,
            im2col_step, remove_center):

        forward_d_stride, forward_block_thread = 8, 1
        backward_d_stride, backward_block_thread = 8, 1

        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.group = group
        ctx.group_channels = group_channels
        ctx.offset_scale = offset_scale
        ctx.im2col_step = im2col_step
        ctx.remove_center = remove_center
        ctx.backward_d_stride = backward_d_stride
        ctx.backward_block_thread = backward_block_thread

        args = [
            input, offset_mask, kernel_h,
            kernel_w, stride_h, stride_w, pad_h,
            pad_w, dilation_h, dilation_w, group,
            group_channels, offset_scale,
            ctx.im2col_step,
            remove_center,
            forward_d_stride,
            forward_block_thread,
            False,
        ]

        output = xav_dsal.dcnv4_forward(*args)
        ctx.save_for_backward(input, offset_mask)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, offset_mask = ctx.saved_tensors

        args = [
            input, offset_mask, ctx.kernel_h,
            ctx.kernel_w, ctx.stride_h, ctx.stride_w, ctx.pad_h,
            ctx.pad_w, ctx.dilation_h, ctx.dilation_w, ctx.group,
            ctx.group_channels, ctx.offset_scale, ctx.im2col_step,
            grad_output.contiguous(), ctx.remove_center,
            ctx.backward_d_stride, ctx.backward_block_thread,
            False
        ]

        grad_input, grad_offset_mask = \
            xav_dsal.dcnv4_backward(*args)

        return grad_input, grad_offset_mask, \
            None, None, None, None, None, None, None,\
            None, None, None, None, None, None

torch.manual_seed(0)
class TestDCNv4(unittest.TestCase):
    # input:  (B, Hin, Win, G * D)
    # offset: (B, Hout, Wout, G * k*k*3)
    # output: (B, Hout, Wout, G * D)
    # D must be a multiple of 8
    B, G, Hin, Win, D = 1, 4, 18, 12, 16
    k, pad, stride, dila = 3, 0, 1, 0
    Hout = (Hin + 2 * pad - ((k-1)*dila+1)) // stride + 1
    Wout = (Win + 2 * pad - ((k-1)*dila+1)) // stride + 1
    input = torch.randn(B, Hin, Win, G*D, dtype=torch.float32)
    offset = torch.randn(B, Hout, Wout, G*k*k*3, dtype=torch.float32)

    def _test_forward(self, device):
        device = torch.device(device)
        input = self.input.to(device).detach().requires_grad_()
        offset = self.offset.to(device).detach().requires_grad_()

        print("input", input.shape)
        print("offset", offset.shape)
        
        out = DCNv4Function.apply(
            input, offset, self.k, self.k, self.stride, self.stride,
            self.pad, self.pad, self.dila, self.dila, self.G, self.D, 1., 1, False)
        
        loss = out.sum()
        loss.backward()

        grad_input = input.grad
        grad_offset = offset.grad

        print(device, "out", out.shape, out)
        print(device, "grad_input", grad_input.shape, grad_input)
        print(device, "grad_offset", grad_offset.shape, grad_offset)
        return [out, grad_input, grad_offset]
    
    def test_cpu(self):
        cpu_result = self._test_forward('cpu')
        xpu_result = self._test_forward('cuda')
        np.testing.assert_allclose(cpu_result[0].cpu().detach().numpy(),
                                   xpu_result[0].cpu().detach().numpy(),
                                   rtol=1e-3, atol=1e-3)
        
        np.testing.assert_allclose(cpu_result[1].cpu().detach().numpy(),
                                   xpu_result[1].cpu().detach().numpy(),
                                   rtol=1e-3, atol=1e-3)
        
        np.testing.assert_allclose(cpu_result[2].cpu().detach().numpy(),
                                   xpu_result[2].cpu().detach().numpy(),
                                   rtol=1e-3, atol=1e-3)

if __name__ == "__main__" : 
    unittest.main()