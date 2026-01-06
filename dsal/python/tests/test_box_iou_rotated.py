# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

import xav_dsal
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, IS_NPU_AVAILABLE


def box_iou_rotated(bboxes1: torch.Tensor,
                    bboxes2: torch.Tensor,
                    mode: str = 'iou',
                    aligned: bool = False,
                    clockwise: bool = True) -> torch.Tensor:
    """Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in
    (x_center, y_center, width, height, angle) format.

    If ``aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    .. note::
        The operator assumes:

        1) The positive direction along x axis is left -> right.

        2) The positive direction along y axis is top -> down.

        3) The w border is in parallel with x axis when angle = 0.

        However, there are 2 opposite definitions of the positive angular
        direction, clockwise (CW) and counter-clockwise (CCW). MMCV supports
        both definitions and uses CW by default.

        Please set ``clockwise=False`` if you are using the CCW definition.

        The coordinate system when ``clockwise`` is ``True`` (default)

            .. code-block:: none

                0-------------------> x (0 rad)
                |  A-------------B
                |  |             |
                |  |     box     h
                |  |   angle=0   |
                |  D------w------C
                v
                y (pi/2 rad)

            In such coordination system the rotation matrix is

            .. math::
                \\begin{pmatrix}
                \\cos\\alpha & -\\sin\\alpha \\\\
                \\sin\\alpha & \\cos\\alpha
                \\end{pmatrix}

            The coordinates of the corner point A can be calculated as:

            .. math::
                P_A=
                \\begin{pmatrix} x_A \\\\ y_A\\end{pmatrix}
                =
                \\begin{pmatrix} x_{center} \\\\ y_{center}\\end{pmatrix} +
                \\begin{pmatrix}\\cos\\alpha & -\\sin\\alpha \\\\
                \\sin\\alpha & \\cos\\alpha\\end{pmatrix}
                \\begin{pmatrix} -0.5w \\\\ -0.5h\\end{pmatrix} \\\\
                =
                \\begin{pmatrix} x_{center}-0.5w\\cos\\alpha+0.5h\\sin\\alpha
                \\\\
                y_{center}-0.5w\\sin\\alpha-0.5h\\cos\\alpha\\end{pmatrix}


        The coordinate system when ``clockwise`` is ``False``

            .. code-block:: none

                0-------------------> x (0 rad)
                |  A-------------B
                |  |             |
                |  |     box     h
                |  |   angle=0   |
                |  D------w------C
                v
                y (-pi/2 rad)

            In such coordination system the rotation matrix is

            .. math::
                \\begin{pmatrix}
                \\cos\\alpha & \\sin\\alpha \\\\
                -\\sin\\alpha & \\cos\\alpha
                \\end{pmatrix}

            The coordinates of the corner point A can be calculated as:

            .. math::
                P_A=
                \\begin{pmatrix} x_A \\\\ y_A\\end{pmatrix}
                =
                \\begin{pmatrix} x_{center} \\\\ y_{center}\\end{pmatrix} +
                \\begin{pmatrix}\\cos\\alpha & \\sin\\alpha \\\\
                -\\sin\\alpha & \\cos\\alpha\\end{pmatrix}
                \\begin{pmatrix} -0.5w \\\\ -0.5h\\end{pmatrix} \\\\
                =
                \\begin{pmatrix} x_{center}-0.5w\\cos\\alpha-0.5h\\sin\\alpha
                \\\\
                y_{center}+0.5w\\sin\\alpha-0.5h\\cos\\alpha\\end{pmatrix}

    Args:
        boxes1 (torch.Tensor): rotated bboxes 1. It has shape (N, 5),
            indicating (x, y, w, h, theta) for each row. Note that theta is in
            radian.
        boxes2 (torch.Tensor): rotated bboxes 2. It has shape (M, 5),
            indicating (x, y, w, h, theta) for each row. Note that theta is in
            radian.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).
        clockwise (bool): flag indicating whether the positive angular
            orientation is clockwise. default True.
            `New in version 1.4.3.`

    Returns:
        torch.Tensor: Return the ious betweens boxes. If ``aligned`` is
        ``False``, the shape of ious is (N, M) else (N,).
    """
    assert mode in ['iou', 'iof']
    mode_dict = {'iou': 0, 'iof': 1}
    mode_flag = mode_dict[mode]
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if aligned:
        ious = bboxes1.new_zeros(rows)
    else:
        if bboxes1.device.type == 'mlu':
            ious = bboxes1.new_zeros([rows, cols])
        else:
            ious = bboxes1.new_zeros(rows * cols)
    if not clockwise:
        flip_mat = bboxes1.new_ones(bboxes1.shape[-1])
        flip_mat[-1] = -1
        bboxes1 = bboxes1 * flip_mat
        bboxes2 = bboxes2 * flip_mat
    if bboxes1.device.type == 'npu':
        scale_mat = bboxes1.new_ones(bboxes1.shape[-1])
        scale_mat[-1] = 1.0 / 0.01745329252
        bboxes1 = bboxes1 * scale_mat
        bboxes2 = bboxes2 * scale_mat
    bboxes1 = bboxes1.contiguous()
    bboxes2 = bboxes2.contiguous()
    xav_dsal.box_iou_rotated(
        bboxes1, bboxes2, ious, mode_flag, aligned)
    if not aligned:
        ious = ious.view(rows, cols)
    return ious

torch.random.manual_seed(0)
np.random.seed(0)

class TestBoxIoURotated:
    """
    def test_box_iou_rotated_cpu(self):
        np_boxes1 = np.asarray(
            [[1.0, 1.0, 3.0, 4.0, 0.5], [2.0, 2.0, 3.0, 4.0, 0.6],
             [7.0, 7.0, 8.0, 8.0, 0.4]],
            dtype=np.float32)
        np_boxes2 = np.asarray(
            [[0.0, 2.0, 2.0, 5.0, 0.3], [2.0, 1.0, 3.0, 3.0, 0.5],
             [5.0, 5.0, 6.0, 7.0, 0.4]],
            dtype=np.float32)
        np_expect_ious = np.asarray(
            [[0.3708, 0.4351, 0.0000], [0.1104, 0.4487, 0.0424],
             [0.0000, 0.0000, 0.3622]],
            dtype=np.float32)
        np_expect_ious_aligned = np.asarray([0.3708, 0.4487, 0.3622],
                                            dtype=np.float32)

        boxes1 = torch.from_numpy(np_boxes1)
        boxes2 = torch.from_numpy(np_boxes2)

        # test cw angle definition
        ious = box_iou_rotated(boxes1, boxes2)
        np.testing.assert_allclose(ious.cpu().numpy(), np_expect_ious, rtol=1e-3, atol=1e-3)

        ious = box_iou_rotated(boxes1, boxes2, aligned=True)
        np.testing.assert_allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, rtol=1e-3, atol=1e-3)

        # test ccw angle definition
        boxes1[..., -1] *= -1
        boxes2[..., -1] *= -1
        ious = box_iou_rotated(boxes1, boxes2, clockwise=False)
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-3)

        ious = box_iou_rotated(boxes1, boxes2, aligned=True, clockwise=False)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-3)

    @pytest.mark.parametrize('device', [
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
        pytest.param(
            'mlu',
            marks=pytest.mark.skipif(
                not IS_MLU_AVAILABLE, reason='requires MLU support')),
        pytest.param(
            'npu',
            marks=pytest.mark.skipif(
                not IS_NPU_AVAILABLE, reason='requires NPU support'))
    ])
    def test_box_iou_rotated(self, device):
        np_boxes1 = np.asarray(
            [[1.0, 1.0, 3.0, 4.0, 0.5], [2.0, 2.0, 3.0, 4.0, 0.6],
             [7.0, 7.0, 8.0, 8.0, 0.4]],
            dtype=np.float32)
        np_boxes2 = np.asarray(
            [[0.0, 2.0, 2.0, 5.0, 0.3], [2.0, 1.0, 3.0, 3.0, 0.5],
             [5.0, 5.0, 6.0, 7.0, 0.4]],
            dtype=np.float32)
        np_expect_ious = np.asarray(
            [[0.3708, 0.4351, 0.0000], [0.1104, 0.4487, 0.0424],
             [0.0000, 0.0000, 0.3622]],
            dtype=np.float32)
        np_expect_ious_aligned = np.asarray([0.3708, 0.4487, 0.3622],
                                            dtype=np.float32)

        boxes1 = torch.from_numpy(np_boxes1).to(device)
        boxes2 = torch.from_numpy(np_boxes2).to(device)

        # test cw angle definition
        ious = box_iou_rotated(boxes1, boxes2)
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-3)

        ious = box_iou_rotated(boxes1, boxes2, aligned=True)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-3)

        # test ccw angle definition
        boxes1[..., -1] *= -1
        boxes2[..., -1] *= -1
        ious = box_iou_rotated(boxes1, boxes2, clockwise=False)
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-3)

        ious = box_iou_rotated(boxes1, boxes2, aligned=True, clockwise=False)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-3)

    def test_box_iou_rotated_iof_cpu(self):
        np_boxes1 = np.asarray(
            [[1.0, 1.0, 3.0, 4.0, 0.5], [2.0, 2.0, 3.0, 4.0, 0.6],
             [7.0, 7.0, 8.0, 8.0, 0.4]],
            dtype=np.float32)
        np_boxes2 = np.asarray(
            [[0.0, 2.0, 2.0, 5.0, 0.3], [2.0, 1.0, 3.0, 3.0, 0.5],
             [5.0, 5.0, 6.0, 7.0, 0.4]],
            dtype=np.float32)
        np_expect_ious = np.asarray(
            [[0.4959, 0.5306, 0.0000], [0.1823, 0.5420, 0.1832],
             [0.0000, 0.0000, 0.4404]],
            dtype=np.float32)
        np_expect_ious_aligned = np.asarray([0.4959, 0.5420, 0.4404],
                                            dtype=np.float32)

        boxes1 = torch.from_numpy(np_boxes1)
        boxes2 = torch.from_numpy(np_boxes2)

        # test cw angle definition
        ious = box_iou_rotated(boxes1, boxes2, mode='iof')
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-3)
        ious = box_iou_rotated(boxes1, boxes2, mode='iof', aligned=True)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-3)

        # test ccw angle definition
        boxes1[..., -1] *= -1
        boxes2[..., -1] *= -1
        ious = box_iou_rotated(boxes1, boxes2, mode='iof', clockwise=False)
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-3)
        ious = box_iou_rotated(
            boxes1, boxes2, mode='iof', aligned=True, clockwise=False)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-3)

    @pytest.mark.parametrize('device', [
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
        pytest.param(
            'mlu',
            marks=pytest.mark.skipif(
                not IS_MLU_AVAILABLE, reason='requires MLU support')),
        pytest.param(
            'npu',
            marks=pytest.mark.skipif(
                not IS_NPU_AVAILABLE, reason='requires NPU support'))
    ])
    def test_box_iou_rotated_iof(self, device):
        np_boxes1 = np.asarray(
            [[1.0, 1.0, 3.0, 4.0, 0.5], [2.0, 2.0, 3.0, 4.0, 0.6],
             [7.0, 7.0, 8.0, 8.0, 0.4]],
            dtype=np.float32)
        np_boxes2 = np.asarray(
            [[0.0, 2.0, 2.0, 5.0, 0.3], [2.0, 1.0, 3.0, 3.0, 0.5],
             [5.0, 5.0, 6.0, 7.0, 0.4]],
            dtype=np.float32)
        np_expect_ious = np.asarray(
            [[0.4959, 0.5306, 0.0000], [0.1823, 0.5420, 0.1832],
             [0.0000, 0.0000, 0.4404]],
            dtype=np.float32)
        np_expect_ious_aligned = np.asarray([0.4959, 0.5420, 0.4404],
                                            dtype=np.float32)

        boxes1 = torch.from_numpy(np_boxes1).to(device)
        boxes2 = torch.from_numpy(np_boxes2).to(device)

        # test cw angle definition
        ious = box_iou_rotated(boxes1, boxes2, mode='iof')
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-3)

        ious = box_iou_rotated(boxes1, boxes2, mode='iof', aligned=True)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-3)

        # test ccw angle definition
        boxes1[..., -1] *= -1
        boxes2[..., -1] *= -1
        ious = box_iou_rotated(boxes1, boxes2, mode='iof', clockwise=False)
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-3)

        ious = box_iou_rotated(
            boxes1, boxes2, mode='iof', aligned=True, clockwise=False)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-3)
    """
    

    def test_box_iou_rotated_random(self):
        for _ in range(3):
            random_len_1 = np.random.randint(10, 2047)
            np_boxes1 = np.random.rand(random_len_1, 5)
            random_len_2 = np.random.randint(10, 2047)
            np_boxes2 = np.random.rand(random_len_2, 5)
            np_boxes1[:, :4] = random_len_1 * np_boxes1[:, :4]
            np_boxes2[:, :4] = random_len_2 * np_boxes2[:, :4]
            np_boxes1[:, 4] = np.arange(random_len_1) / 180 * np.pi
            np_boxes2[:, 4] = np.arange(random_len_2) / 180 * np.pi

            boxes1 = torch.from_numpy(np_boxes1).to(torch.float32) 
            boxes2 = torch.from_numpy(np_boxes2).to(torch.float32) 
            boxes1_gpu = boxes1.cuda()
            boxes2_gpu = boxes2.cuda()
            
            # box_iou_rotated
            ious_cpu = box_iou_rotated(boxes1, boxes2)
            ious_gpu = box_iou_rotated(boxes1_gpu, boxes2_gpu)
            assert np.allclose(ious_cpu.numpy(), ious_gpu.cpu().numpy(), atol=1e-3, rtol=1e-3)
   
            # box_iou_rotated iof
            ious_cpu = box_iou_rotated(boxes1, boxes2, mode='iof')
            ious_gpu = box_iou_rotated(boxes1_gpu, boxes2_gpu, mode='iof')
            assert np.allclose(ious_cpu.numpy(), ious_gpu.cpu().numpy(), atol=1e-3, rtol=1e-3)
            
           
    def test_box_iou_rotated_random_align(self):
        for _ in range(3):
            random_len = np.random.randint(10, 2047)
            np_boxes1 = np.random.rand(random_len, 5)
            np_boxes2 = np.random.rand(random_len, 5)
            np_boxes1[:, :4] = random_len * np_boxes1[:, :4]
            np_boxes2[:, :4] = random_len * np_boxes2[:, :4]
            np_boxes1[:, 4] = np.arange(random_len) / 180 * np.pi
            np_boxes2[:, 4] = np.arange(random_len) / 180 * np.pi

            boxes1 = torch.from_numpy(np_boxes1).to(torch.float32) 
            boxes2 = torch.from_numpy(np_boxes2).to(torch.float32) 
            boxes1_gpu = boxes1.cuda()
            boxes2_gpu = boxes2.cuda()
            
            # box_iou_rotated
            ious_cpu = box_iou_rotated(boxes1, boxes2)
            ious_gpu = box_iou_rotated(boxes1_gpu, boxes2_gpu)
            assert np.allclose(ious_cpu.numpy(), ious_gpu.cpu().numpy(), atol=1e-3, rtol=1e-3)
            
            # box_iou_rotated aligned
            ious_cpu = box_iou_rotated(boxes1, boxes2, aligned=True)
            ious_gpu = box_iou_rotated(boxes1_gpu, boxes2_gpu, aligned=True)
            assert np.allclose(ious_cpu.numpy(), ious_gpu.cpu().numpy(), atol=1e-3, rtol=1e-3)
            
            # box_iou_rotated iof
            ious_cpu = box_iou_rotated(boxes1, boxes2, mode='iof')
            ious_gpu = box_iou_rotated(boxes1_gpu, boxes2_gpu, mode='iof')
            assert np.allclose(ious_cpu.numpy(), ious_gpu.cpu().numpy(), atol=1e-3, rtol=1e-3)
            
            # box_iou_rotated iof aligned
            ious_cpu = box_iou_rotated(boxes1, boxes2, mode='iof', aligned=True)
            ious_gpu = box_iou_rotated(boxes1_gpu, boxes2_gpu, mode='iof', aligned=True)
            assert np.allclose(ious_cpu.numpy(), ious_gpu.cpu().numpy(), atol=1e-3, rtol=1e-3)
            
            