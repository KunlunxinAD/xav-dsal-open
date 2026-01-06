# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from typing import Optional, Union
import torch
import numpy as np
import pytest

# import sigmoid_focal_loss_ext
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext('_ext', [
    'sigmoid_focal_loss_forward', 'sigmoid_focal_loss_backward',
])


inputs = [
    ([[1.0, 0], [0, 1.0]], [0, 1]),
    ([[1.0, 0, -1.0], [0, 1.0, 2.0]], [2, 1]),
    ([[1e-6, 2e-6, 3e-6], [4e-6, 5e-5, 6e-4], [7e-3, 8e-2, 9e-1]], [1, 2, 0]),
]

sigmoid_outputs = [
    (0.13562961, [[-0.00657264, 0.11185755], [0.11185755, -0.00657264]]),
    (
        1.10251057,
        [[0.28808805, 0.11185755, -0.09602935], [0.11185755, -0.00657264, 0.40376765]],
    ),
    (
        0.42287254,
        [
            [0.07457182, -0.02485716, 0.07457201],
            [0.07457211, 0.07457669, -0.02483728],
            [-0.02462499, 0.08277918, 0.18050370],
        ],
    ),
]


class SigmoidFocalLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        target: Union[torch.LongTensor, torch.cuda.LongTensor],
        weight: torch.Tensor,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert target.dtype == torch.long
        assert input.dim() == 2
        assert target.dim() == 1
        assert input.size(0) == target.size(0)
        ctx.reduction_dict = {'none': 0, 'mean': 1, 'sum': 2}
        assert reduction in ctx.reduction_dict.keys()

        ctx.gamma = float(gamma)
        ctx.alpha = float(alpha)
        ctx.reduction = ctx.reduction_dict[reduction]

        output = input.new_zeros(input.size())

        ext_module.sigmoid_focal_loss_forward(
            input, target, weight, output, ctx.gamma, ctx.alpha)
        if ctx.reduction == ctx.reduction_dict['mean']:
            output = output.sum() / input.size(0)
        elif ctx.reduction == ctx.reduction_dict['sum']:
            output = output.sum()
        ctx.save_for_backward(input, target, weight)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        input, target, weight = ctx.saved_tensors

        grad_input = input.new_zeros(input.size())

        ext_module.sigmoid_focal_loss_backward(
            input,
            target,
            weight,
            grad_input,
            ctx.gamma,
            ctx.alpha)

        grad_input *= grad_output
        if ctx.reduction == ctx.reduction_dict['mean']:
            grad_input /= input.size(0)
        return grad_input, None, None, None, None, None


class Testfocalloss:
    def _test_sigmoid(self, dtype=torch.float):
        alpha = 0.25
        gamma = 2.0
        for case, output in zip(inputs, sigmoid_outputs):
            np_x = np.array(case[0])
            np_y = np.array(case[1])
            np_x_grad = np.array(output[1])

            x = torch.from_numpy(np_x).cuda().type(dtype)
            x.requires_grad_()
            y = torch.from_numpy(np_y).cuda().long()
            loss = torch.zeros_like(x).cuda()
            loss = SigmoidFocalLossFunction.apply(x, y, torch.empty(0), gamma, alpha)
            loss.backward()
            assert np.allclose(loss.data.cpu().numpy(), output[0], 1e-2)
            assert np.allclose(x.grad.data.cpu(), np_x_grad, 1e-2)

    def test_sigmoid_float(self):
        self._test_sigmoid(dtype=torch.float)


if __name__ == "__main__":
    unittest.main()
