# Copyright (c) OpenMMLab. All rights reserved.
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
from data_compare import compare_tensors

try:
    # If PyTorch version >= 1.6.0 and fp16 is enabled, torch.cuda.amp.autocast
    # would be imported and used; we should test if our modules support it.
    from torch.cuda.amp import autocast
except ImportError:
    pass

cur_dir = os.path.dirname(os.path.abspath(__file__))

input_t = [[[[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]]]
output_t = [[[[0.5, 1.5, 2.5, 1.5], [1.0, 3.0, 5.0, 3.0], [1.0, 3.0, 5.0, 3.0],
              [0.5, 1.5, 2.5, 1.5]]]]
input_grad = [[[[2., 2., 2.], [2., 2., 2.], [2., 2., 2.]]]]
dcn_w_grad = [[[[9., 9.], [9., 9.]]]]
dcn_offset_w_grad = [[[[-7.0, -4.0], [0.0, 0.0]]], [[[-9.0, 7.5], [-6.0,
                                                                   5.0]]],
                     [[[-4.0, -7.0], [0.0, 0.0]]],
                     [[[-7.5, -9.0], [-5.0, -6.0]]],
                     [[[-7.0, -4.0], [-7.0, -4.0]]],
                     [[[-6.0, 5.0], [-9.0, 7.5]]],
                     [[[-4.0, -7.0], [-4.0, -7.0]]],
                     [[[-5.0, -6.0], [-7.5, -9.0]]], [[[10.5, 6.0], [7.0,
                                                                     4.0]]],
                     [[[6.0, 10.5], [4.0, 7.0]]], [[[7.0, 4.0], [10.5, 6.0]]],
                     [[[4.0, 7.0], [6.0, 10.5]]]]
dcn_offset_b_grad = [
    -3.0, -1.5, -3.0, -1.5, -3.0, -1.5, -3.0, -1.5, 4.5, 4.5, 4.5, 4.5
]

class ModulatedDeformConv2dFunction(Function):

    @staticmethod
    def symbolic(g, input, offset, mask, weight, bias, stride, padding,
                 dilation, groups, deform_groups):
        input_tensors = [input, offset, mask, weight]
        if bias is not None:
            input_tensors.append(bias)
        return g.op(
            'mmcv::MMCVModulatedDeformConv2d',
            *input_tensors,
            stride_i=stride,
            padding_i=padding,
            dilation_i=dilation,
            groups_i=groups,
            deform_groups_i=deform_groups)

    @staticmethod
    def _calculate_sort_index(kernel_h, kernel_w, deformable_group):
        split_num = deformable_group * 2 * kernel_h * kernel_w
        sort_index = list(range(split_num))
        sort_index_fp = (sort_index[1::2] + sort_index[::2])
        sort_index_bp_dict = {i: idx for idx, i in enumerate(sort_index_fp)}
        sort_index_bp = [sort_index_bp_dict[i] for i in sort_index]
        sort_index_fp = torch.IntTensor(sort_index_fp)
        sort_index_bp = torch.IntTensor(sort_index_bp)
        sort_index_fp = sort_index_fp.npu()
        sort_index_bp = sort_index_bp.npu()
        return sort_index_fp, sort_index_bp

    @staticmethod
    def _npu_forward(ctx, input_tensor, offset, mask, weight, bias):
        _, _, kernel_h, kernel_w = weight.shape
        conv2d_bias = bias if len(bias) > 0 else None
        sort_index_fp, sort_index_bp = \
            ModulatedDeformConv2dFunction._calculate_sort_index(
                kernel_w, kernel_h, ctx.deform_groups)
        select_offset = offset.index_select(1, sort_index_fp)
        offset_all = torch.cat([select_offset, mask], dim=1)
        output, offset_out = torch.npu_deformable_conv2d(
            input_tensor,
            weight,
            offset_all,
            conv2d_bias,
            kernel_size=[kernel_w, kernel_h],
            stride=[1, 1, ctx.stride[0], ctx.stride[1]],
            padding=[
                ctx.padding[0], ctx.padding[0], ctx.padding[1], ctx.padding[1]
            ],
            dilation=[1, 1, ctx.dilation[0], ctx.dilation[1]],
            groups=ctx.groups,
            deformable_groups=ctx.deform_groups,
            modulated=True)
        if weight.requires_grad or mask.requires_grad or offset.requires_grad \
                or input_tensor.requires_grad:
            ctx.save_for_backward(input_tensor, weight, offset_out, offset_all,
                                  sort_index_bp)
        return output

    @staticmethod
    def _npu_backward(ctx, grad_output):
        input_tensor, weight, offset_out, offset_all, sort_index_bp = \
            ctx.saved_tensors
        grad_input, grad_weight, grad_offset_all, grad_bias = \
            torch.npu_deformable_conv2dbk(
                input_tensor, grad_output, offset_out, weight, offset_all,
                kernel_size=[weight.shape[3], weight.shape[2]],
                stride=[1, 1, ctx.stride[0], ctx.stride[1]],
                padding=[ctx.padding[0], ctx.padding[0], ctx.padding[1],
                         ctx.padding[1]],
                dilation=[1, 1, ctx.dilation[0], ctx.dilation[1]],
                groups=ctx.groups, deformable_groups=ctx.deform_groups,
                modulated=True)
        grad_offset = grad_offset_all.index_select(1, sort_index_bp)
        grad_mask = grad_offset_all[:, grad_offset.shape[1]:, :, :]
        if not ctx.with_bias:
            grad_bias = None
        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None, None, None, None)

    @staticmethod
    def forward(ctx,
                input: torch.Tensor,
                offset: torch.Tensor,
                mask: torch.Tensor,
                weight: nn.Parameter,
                bias: Optional[nn.Parameter] = None,
                stride: int = 1,
                padding: int = 0,
                dilation: int = 1,
                groups: int = 1,
                deform_groups: int = 1) -> torch.Tensor:
        if input is not None and input.dim() != 4:
            raise ValueError(
                f'Expected 4D tensor as input, got {input.dim()}D tensor \
                  instead.')
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deform_groups = deform_groups
        ctx.with_bias = bias is not None
        ctx.device = input.device.type
        if not ctx.with_bias:
            bias = input.new_empty(0)  # fake tensor
        # When pytorch version >= 1.6.0, amp is adopted for fp16 mode;
        # amp won't cast the type of model (float32), but "offset" is cast
        # to float16 by nn.Conv2d automatically, leading to the type
        # mismatch with input (when it is float32) or weight.
        # The flag for whether to use fp16 or amp is the type of "offset",
        # we cast weight and input to temporarily support fp16 and amp
        # whatever the pytorch version is.
        input = input.type_as(offset)
        weight = weight.type_as(input)
        bias = bias.type_as(input)  # type: ignore
        mask = mask.type_as(input)
        if ctx.device == 'npu':
            output = ModulatedDeformConv2dFunction._npu_forward(
                ctx, input, offset, mask, weight, bias)
            return output
        ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(
            ModulatedDeformConv2dFunction._output_size(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        xav_dsal.modulated_deform_conv_forward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            output,
            ctx._bufs[1],
            kernel_h=weight.size(2),
            kernel_w=weight.size(3),
            stride_h=ctx.stride[0],
            stride_w=ctx.stride[1],
            pad_h=ctx.padding[0],
            pad_w=ctx.padding[1],
            dilation_h=ctx.dilation[0],
            dilation_w=ctx.dilation[1],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            with_bias=ctx.with_bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        if ctx.device == 'npu':
            return ModulatedDeformConv2dFunction._npu_backward(
                ctx, grad_output)
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        grad_output = grad_output.contiguous()
        xav_dsal.modulated_deform_conv_backward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            ctx._bufs[1],
            grad_input,
            grad_weight,
            grad_bias,
            grad_offset,
            grad_mask,
            grad_output,
            kernel_h=weight.size(2),
            kernel_w=weight.size(3),
            stride_h=ctx.stride[0],
            stride_w=ctx.stride[1],
            pad_h=ctx.padding[0],
            pad_w=ctx.padding[1],
            dilation_h=ctx.dilation[0],
            dilation_w=ctx.dilation[1],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            with_bias=ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None)

    @staticmethod
    def _output_size(ctx, input, weight):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = ctx.padding[d]
            kernel = ctx.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = ctx.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                'convolution input is too small (output would be ' +
                'x'.join(map(str, output_size)) + ')')
        return output_size


modulated_deform_conv2d = ModulatedDeformConv2dFunction.apply


class ModulatedDeformConv2d(nn.Module):

    # @deprecated_api_warning({'deformable_groups': 'deform_groups'},
    #                         cls_name='ModulatedDeformConv2d')
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 deform_groups: int = 1,
                 bias: Union[bool, str] = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups,
                         *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x: torch.Tensor, offset: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


class ModulatedDeformConv2dPack(ModulatedDeformConv2d):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)
        self.init_weights()

    def init_weights(self) -> None:
        super().init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            # the key is different in early versions
            # In version < 2, ModulatedDeformConvPack
            # loads previous benchmark models.
            if (prefix + 'conv_offset.weight' not in state_dict
                    and prefix[:-1] + '_offset.weight' in state_dict):
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(
                    prefix[:-1] + '_offset.weight')
            if (prefix + 'conv_offset.bias' not in state_dict
                    and prefix[:-1] + '_offset.bias' in state_dict):
                state_dict[prefix +
                           'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
                                                                '_offset.bias')

        if version is not None and version > 1:
            print_log(
                f'ModulatedDeformConvPack {prefix.rstrip(".")} is upgraded to '
                'version 2.',
                logger='root')

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)


class TestCustomModulatedDeformConv(unittest.TestCase):
    def _test_mdconv(test_case, dtype=torch.float):
        input = torch.tensor(
            input_t, dtype=dtype, device='cuda').requires_grad_()

        dcn = ModulatedDeformConv2dPack(
            1,
            1,
            kernel_size=(
                2,
                2,
            ),
            stride=1,
            padding=1,
            deform_groups=1,
            bias=False,
        ).to('cuda')

        dcn.weight.data.fill_(1.0)
        dcn.type(dtype)
        output = dcn(input)

        test_case.assertTrue(
            np.allclose(
                output.cpu().detach().numpy(),
                output_t,
                1e-3,
            )
        )

        output.sum().backward()
        test_case.assertTrue(
            np.allclose(
                input.grad.cpu().detach().numpy(),
                input_grad,
                1e-3,
            )
        )
        test_case.assertTrue(
            np.allclose(
                dcn.weight.grad.cpu().detach().numpy(),
                dcn_w_grad,
                1e-3,
            )
        )

        test_case.assertTrue(
            np.allclose(
                dcn.conv_offset.weight.grad.cpu().detach().numpy(),
                dcn_offset_w_grad,
                1e-3,
            ),
            f"{dcn.conv_offset.weight.grad.cpu().detach().numpy()} vs {dcn_offset_w_grad}",
        )
        test_case.assertTrue(
            np.allclose(
                dcn.conv_offset.bias.grad.cpu().detach().numpy(),
                dcn_offset_b_grad,
                1e-3,
            ),
            f"{dcn.conv_offset.bias.grad.cpu().detach().numpy()} vs {dcn_offset_b_grad}",
        )

    def _test_mdconv_cpu(test_case, dtype=torch.float):
        input = torch.tensor(
            input_t, dtype=dtype, device='cpu').requires_grad_()

        dcn = ModulatedDeformConv2dPack(
            1,
            1,
            kernel_size=(
                2,
                2,
            ),
            stride=1,
            padding=1,
            deform_groups=1,
            bias=False,
        ).to('cpu')

        dcn.weight.data.fill_(1.0)
        dcn.type(dtype)
        output = dcn(input)
        test_case.assertTrue(
            np.allclose(
                output.cpu().detach().numpy(),
                output_t,
                1e-3,
            )
        )

        output.sum().backward()
        test_case.assertTrue(
            np.allclose(
                input.grad.cpu().detach().numpy(),
                input_grad,
                1e-3,
            )
        )
        test_case.assertTrue(
            np.allclose(
                dcn.weight.grad.cpu().detach().numpy(),
                dcn_w_grad,
                1e-3,
            )
        )
        test_case.assertTrue(
            np.allclose(
                dcn.conv_offset.weight.grad.cpu().detach().numpy(),
                dcn_offset_w_grad,
                1e-3,
            ),
            f"{dcn.conv_offset.weight.grad.cpu().detach().numpy()} vs {dcn_offset_w_grad}",
        )
        test_case.assertTrue(
            np.allclose(
                dcn.conv_offset.bias.grad.cpu().detach().numpy(),
                dcn_offset_b_grad,
                1e-3,
            ),
            f"{dcn.conv_offset.bias.grad.cpu().detach().numpy()} vs {dcn_offset_b_grad}",
        )


    def test_custom_modulated_deform_conv(self):
        self._test_mdconv_cpu(torch.float)
        self._test_mdconv(torch.float)

def modulated_deform_conv_fwd(file_path, test_id, data_type, device, save, compare):
    print(f"---- test fwd, test_id {test_id}, data_type {data_type}, device {device}, save {save}, compare {compare}")
    data = torch.load(f"{file_path}/deform_conv_fwd_{test_id}.pt", weights_only=True)
    output = torch.load(f"./data/test_modulated_deform_conv/fwd_output_fp32_{test_id}.pt", weights_only=True)
    output_test = torch.zeros_like(data['output'], dtype=data_type, device=device)

    xav_dsal.modulated_deform_conv_forward(
            data['input'].to(data_type).to(device),
            data['weight'].to(data_type).to(device),
            data['bias'].to(data_type).to(device),
            data['ctx0'].to(data_type).to(device),
            data['offset'].to(data_type).to(device),
            data['mask'].to(data_type).to(device),
            output_test,
            data['ctx1'].to(data_type).to(device),
            data['kernel_h'],
            data['kernel_w'],
            data['stride_h'],
            data['stride_w'],
            data['pad_h'],
            data['pad_w'],
            data['dilation_h'],
            data['dilation_w'],
            data['group'],
            data['deformable_group'],
            data['with_bias'])
    if data_type == torch.float16:
        rtol, atol = 5e-3, 5e-3
    else:
        rtol, atol = 1e-5, 1e-5

    if save:
        torch.save(output_test, f"./data/test_modulated_deform_conv/fwd_output_{test_id}.pt")
    if compare:
        compare_tensors(output_test.cpu(), output.cpu().to(data_type), rtol, atol, "output")

def modulated_deform_conv_bwd(file_path, test_id, data_type, device, save, compare):
    print(f"---- test bwd, test_id {test_id}, data_type {data_type}, device {device}, save {save}, compare {compare}")
    data = torch.load(f"{file_path}/deform_conv_bwd_{test_id}.pt", weights_only=True)
    output = torch.load(f"./data/test_modulated_deform_conv/bwd_output_fp32_{test_id}.pt", weights_only=True)

    grad_input = torch.zeros_like(data['grad_input'], dtype=data_type, device=device)
    grad_weight = torch.zeros_like(data['grad_weight'], dtype=data_type, device=device)
    grad_bias = torch.zeros_like(data['grad_bias'], dtype=data_type, device=device)
    grad_offset = torch.zeros_like(data['grad_offset'], dtype=data_type, device=device)
    grad_mask = torch.zeros_like(data['grad_mask'], dtype=data_type, device=device)

    xav_dsal.modulated_deform_conv_backward(
            data['input'].to(data_type).to(device),
            data['weight'].to(data_type).to(device),
            data['bias'].to(data_type).to(device),
            data['ctx0'].to(data_type).to(device),
            data['offset'].to(data_type).to(device),
            data['mask'].to(data_type).to(device),
            data['ctx1'].to(data_type).to(device),
            grad_input,
            grad_weight,
            grad_bias,
            grad_offset,
            grad_mask,
            data['grad_output'].to(data_type).to(device),
            data['kernel_h'],
            data['kernel_w'],
            data['stride_h'],
            data['stride_w'],
            data['pad_h'],
            data['pad_w'],
            data['dilation_h'],
            data['dilation_w'],
            data['group'],
            data['deformable_group'],
            data['with_bias'])
            
    if data_type == torch.float16:
        rtol, atol = 5e-3, 5e-3
    else:
        rtol, atol = 1e-5, 1e-5
    if compare:
        assert(compare_tensors(grad_input.cpu(), output['grad_input'].cpu().to(data_type), rtol, atol, "grad_input"))
        assert(compare_tensors(grad_weight.cpu(), output['grad_weight'].cpu().to(data_type), rtol, atol, "grad_weight"))
        assert(compare_tensors(grad_bias.cpu(), output['grad_bias'].cpu().to(data_type), rtol, atol, "grad_bias"))
        assert(compare_tensors(grad_offset.cpu(), output['grad_offset'].cpu().to(data_type), rtol, atol, "grad_offset"))
        assert(compare_tensors(grad_mask.cpu(), output['grad_mask'].cpu().to(data_type), rtol, atol, "grad_mask"))
    if save:
        save_data = {}
        save_data['grad_input'] = grad_input
        save_data['grad_weight'] = grad_weight
        save_data['grad_bias'] = grad_bias
        save_data['grad_offset'] = grad_offset
        save_data['grad_mask'] = grad_mask
        torch.save(save_data, f"./data/test_modulated_deform_conv/bwd_output_{test_id}.pt")

@pytest.fixture
def base_path():
    return "./data/test_modulated_deform_conv"

@pytest.mark.parametrize("test_id", [1, 2])
@pytest.mark.parametrize("device", ["cuda"])
# @pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("data_type", [torch.float16, torch.float32])
def test_modulated_deform_conv(base_path, test_id, data_type, device):
    modulated_deform_conv_fwd(base_path, test_id, data_type, device, False, True)
    modulated_deform_conv_bwd(base_path, test_id, data_type, device, False, True)

if __name__ == "__main__":
    # unittest.main()
    pytest.main(["-v", "-s", "test_modulated_deform_conv.py"])
