from typing import Any, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import Function


def boxes_iou3d(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Calculate boxes 3D IoU.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (N, 7).
        boxes_b (torch.Tensor): Input boxes b with shape (M, 7).

    Returns:
        torch.Tensor: 3D IoU result with shape (N, M).
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7, \
        'Input boxes shape should be (N, 7)'
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    overlaps_bev = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    overlaps_bev = overlaps_bev.to(boxes_a.device)
    torch.ops.xav_dsal.iou3d_boxes_overlap_bev_forward(boxes_a.contiguous(),
                                             boxes_b.contiguous(),
                                             overlaps_bev)
    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)
    overlaps_3d = overlaps_bev * overlaps_h
    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)
    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)
    return iou3d


def nms3d(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """3D NMS function GPU implementation (for BEV boxes).

    Args:
        boxes (torch.Tensor): Input boxes with the shape of (N, 7)
            ([x, y, z, dx, dy, dz, heading]).
        scores (torch.Tensor): Scores of boxes with the shape of (N).
        iou_threshold (float): Overlap threshold of NMS.

    Returns:
        torch.Tensor: Indexes after NMS.
    """
    assert boxes.size(1) == 7, 'Input boxes shape should be (N, 7)'
    order = scores.sort(0, descending=True)[1]
    boxes = boxes[order].contiguous()

    keep = boxes.new_zeros(boxes.size(0), dtype=torch.long)
    num_out = boxes.new_zeros(size=(), dtype=torch.long)
    torch.ops.xav_dsal.iou3d_nms3d_forward(boxes, keep, num_out, nms_overlap_thresh=iou_threshold)
    keep = order[keep[:num_out].to(boxes.device)].contiguous()
    return keep


def nms3d_normal(boxes: Tensor, scores: Tensor,
                 iou_threshold: float) -> Tensor:
    """Normal 3D NMS function GPU implementation. The overlap of two boxes for
    IoU calculation is defined as the exact overlapping area of the two boxes
    WITH their yaw angle set to 0.

    Args:
        boxes (torch.Tensor): Input boxes with shape (N, 7).
            ([x, y, z, dx, dy, dz, heading]).
        scores (torch.Tensor): Scores of predicted boxes with shape (N).
        iou_threshold (float): Overlap threshold of NMS.

    Returns:
        torch.Tensor: Remaining indices with scores in descending order.
    """
    assert boxes.shape[1] == 7, 'Input boxes shape should be (N, 7)'
    order = scores.sort(0, descending=True)[1]
    boxes = boxes[order].contiguous()

    keep = boxes.new_zeros(boxes.size(0), dtype=torch.long)
    num_out = boxes.new_zeros(size=(), dtype=torch.long)
    torch.ops.xav_dsal.iou3d_nms3d_normal_forward(boxes, keep, num_out, nms_overlap_thresh=iou_threshold)
    return order[keep[:num_out].to(boxes.device)].contiguous()

def boxes_iou_bev(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Calculate boxes BEV IoU.
    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (N, 7).
        boxes_b (torch.Tensor): Input boxes b with shape (M, 7).
    Returns:
        torch.Tensor: BEV IoU result with shape (N, M).
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7, \
        'Input boxes shape should be (N, 7)'
    ans_iou = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    ans_iou = ans_iou.to(boxes_a.device)
    torch.ops.xav_dsal.boxes_iou_bev_gpu(boxes_a.contiguous(),
                                         boxes_b.contiguous(),
                                         ans_iou)
    return ans_iou

def paired_boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N)
    """
    assert boxes_a.shape[0] == boxes_b.shape[0]
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(-1, 1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(-1, 1)

    # bev overlap
    overlaps_bev = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], 1)))  # (N, ``)
    torch.ops.xav_dsal.paired_boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(-1, 1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d.view(-1)