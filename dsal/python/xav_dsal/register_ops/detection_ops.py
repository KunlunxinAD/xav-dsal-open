import torch
from ..registry import register_op

try:
    box_iou_rotated = torch.ops.xav_dsal.box_iou_rotated
    register_op("box_iou_rotated", box_iou_rotated, for_mmcv=True)

    # iou3d_boxes_overlap_bev_forward
    iou3d_boxes_overlap_bev_forward = torch.ops.xav_dsal.iou3d_boxes_overlap_bev_forward
    register_op("iou3d_boxes_overlap_bev_forward", iou3d_boxes_overlap_bev_forward, for_mmcv=True)

    # iou3d_nms3d_forward
    iou3d_nms3d_forward = torch.ops.xav_dsal.iou3d_nms3d_forward
    register_op("iou3d_nms3d_forward", iou3d_nms3d_forward, for_mmcv=True)

    # iou3d_nms3d_normal_forward
    iou3d_nms3d_normal_forward = torch.ops.xav_dsal.iou3d_nms3d_normal_forward
    register_op("iou3d_nms3d_normal_forward", iou3d_nms3d_normal_forward, for_mmcv=True)

    # boxes_iou_bev_gpu
    boxes_iou_bev_gpu = torch.ops.xav_dsal.boxes_iou_bev_gpu
    register_op("boxes_iou_bev_gpu", boxes_iou_bev_gpu, for_mmcv=False)

    # paired_boxes_overlap_bev_gpu
    paired_boxes_overlap_bev_gpu = torch.ops.xav_dsal.paired_boxes_overlap_bev_gpu
    register_op("paired_boxes_overlap_bev_gpu", paired_boxes_overlap_bev_gpu, for_mmcv=False)
except Exception as e:
    raise RuntimeError(f"Failed to load op: {e}")