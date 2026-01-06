# Copyright(c) OpenMMLab.All rights reserved.
import unittest 
from typing import Optional, Union
import torch
import numpy as np
import pytest

from mmcv.utils import ext_loader 
ext_module = ext_loader.load_ext('_ext',
        ['softmax_focal_loss_forward', 'softmax_focal_loss_backward'])

class SoftmaxFocalLossFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                input: torch.Tensor,
                target: Union[torch.LongTensor, torch.cuda.LongTensor],
                gamma: float = 2.0,
                alpha: float = 0.25,
                weight: Optional[torch.Tensor] = None,
                reduction='mean') -> torch.Tensor:

        assert target.dtype == torch.long
        assert input.dim() == 2
        assert target.dim() == 1
        assert input.size(0) == target.size(0)
        if weight is None:
            weight = input.new_empty(0)
        else:
            assert weight.dim() == 1
            assert input.size(1) == weight.size(0)
        ctx.reduction_dict = {'none': 0, 'mean': 1, 'sum': 2}
        assert reduction in ctx.reduction_dict.keys()

        ctx.gamma = float(gamma)
        ctx.alpha = float(alpha)
        ctx.reduction = ctx.reduction_dict[reduction]

        channel_stats, _ = torch.max(input, dim=1)
        input_softmax = input - channel_stats.unsqueeze(1).expand_as(input)
        input_softmax.exp_()

        channel_stats = input_softmax.sum(dim=1)
        input_softmax /= channel_stats.unsqueeze(1).expand_as(input)

        output = input.new_zeros(input.size(0))
        ext_module.softmax_focal_loss_forward(
            input_softmax,
            target,
            weight,
            output,
            ctx.gamma,
            ctx.alpha)

        if ctx.reduction == ctx.reduction_dict['mean']:
            output = output.sum() / input.size(0)
        elif ctx.reduction == ctx.reduction_dict['sum']:
            output = output.sum()
        ctx.save_for_backward(input_softmax, target, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        input_softmax, target, weight = ctx.saved_tensors
        buff = input_softmax.new_zeros(input_softmax.size(0))
        grad_input = input_softmax.new_zeros(input_softmax.size())

        ext_module.softmax_focal_loss_backward(
            input_softmax,
            target,
            weight,
            buff,
            grad_input,
            ctx.gamma,
            ctx.alpha)

        print(input_softmax.shape, "output", grad_input)
        grad_input *= grad_output
        if ctx.reduction == ctx.reduction_dict['mean']:
            grad_input /= input_softmax.size(0)

        return grad_input, None, None, None, None, None

class TestSoftmaxFocalLoss(unittest.TestCase):
    # input dims : (len, n_classes)
    dims =[(64 * 8, 16), (64 * 128, 1), (64 * 4, 256), (64 * 8, 1024), (13 * 7 + 5, 53), (9, 3)] 
    _input =[] 
    _target =[]
    
    def _test_softmax(self, dtype, device) : 
        output =[] 
        for i in range(len(self.dims)) : 
            input = self._input[i].to(device).detach()  # Detach to ensure a new computation graph
            input.requires_grad = True  # Enable gradient computation
            target = self._target[i].to(device)
            loss = SoftmaxFocalLossFunction.apply(input, target, 3, 0.25) 
            loss.backward()
            print(device, self.dims[i], loss, input.grad) 
            output.append([loss, input.grad]) 
        return output
            
    def test_softmax_focal_loss(self):
        #initialize input and target
        dtype = torch.float32 
        for dim in self.dims: 
            self._input.append(torch.randn(* dim, dtype = dtype, requires_grad = True)) 
            self._target.append(torch.randint(low = 0, high = dim[1], size =(dim[0], ), dtype = torch.long))

        output_cpu = self._test_softmax(dtype, 'cpu') 
        output_gpu = self._test_softmax(dtype, 'cuda')
        
        for i in range(len(output_cpu)) : 
            np.testing.assert_allclose(output_cpu[i][0].detach().numpy(), 
                output_gpu[i][0].detach().cpu().numpy(), atol = 1e-3) 

            np.testing.assert_allclose(output_cpu[i][1].detach().numpy(), 
                output_gpu[i][1].detach().cpu().numpy(), atol = 1e-3, rtol = 1e-3, ) 



if __name__ == "__main__" : 
    unittest.main()
