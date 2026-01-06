import pytest
import torch
from torch import Tensor
import os
import pandas as pd
import numpy as np
import math
import xav_dsal
from math import atan2, cos, fabs, sin
from typing import List
from data_compare import get_diff, compare_tensors

EPS = 1e-8


class Point:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def set(self, _x: float, _y: float):
        self.x = _x
        self.y = _y

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Point(x, y)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Point(x, y)


def cross(p1: Point, p2: Point, p0: Point) -> float:
    if p0 is None:
        return p1.x * p2.y - p1.y * p2.x
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)


def check_rect_cross(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    ret = min(p1.x, p2.x) <= max(q1.x, q2.x) and \
          min(q1.x, q2.x) <= max(p1.x, p2.x) and \
          min(p1.y, p2.y) <= max(q1.y, q2.y) and \
          min(q1.y, q2.y) <= max(p1.y, p2.y)
    return ret


def box_overlap(box_a: List[float], box_b: List[float]):
    a_angle = box_a[6]
    b_angle = box_b[6]
    a_dx_half = box_a[3] / 2
    b_dx_half = box_b[3] / 2
    a_dy_half = box_a[4] / 2
    b_dy_half = box_b[4] / 2
    a_x1 = box_a[0] - a_dx_half
    a_y1 = box_a[1] - a_dy_half
    a_x2 = box_a[0] + a_dx_half
    a_y2 = box_a[1] + a_dy_half
    b_x1 = box_b[0] - b_dx_half
    b_y1 = box_b[1] - b_dy_half
    b_x2 = box_b[0] + b_dx_half
    b_y2 = box_b[1] + b_dy_half

    center_a = Point(box_a[0], box_a[1])
    center_b = Point(box_b[0], box_b[1])

    box_a_corners = [Point()] * 5
    box_a_corners[0] = Point(a_x1, a_y1)
    box_a_corners[1] = Point(a_x2, a_y1)
    box_a_corners[2] = Point(a_x2, a_y2)
    box_a_corners[3] = Point(a_x1, a_y2)

    box_b_corners = [Point()] * 5
    box_b_corners[0] = Point(b_x1, b_y1)
    box_b_corners[1] = Point(b_x2, b_y1)
    box_b_corners[2] = Point(b_x2, b_y2)
    box_b_corners[3] = Point(b_x1, b_y2)
    # get oriented corners
    a_angle_cos = cos(a_angle)
    a_angle_sin = sin(a_angle)

    b_angle_cos = cos(b_angle)
    b_angle_sin = sin(b_angle)
    for k in range(4):
        rotate_point_a = rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k])
        box_a_corners[k] = rotate_point_a
        rotate_point_b = rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k])
        box_b_corners[k] = rotate_point_b
    box_a_corners[4] = box_a_corners[0]
    box_b_corners[4] = box_b_corners[0]
    cross_points = [Point()] * 16
    poly_center = Point(0, 0)
    cnt = 0
    flag = 0
    for i in range(4):
        for j in range(4):
            flag, ans_point = intersection(box_a_corners[i + 1], box_a_corners[i],
                                           box_b_corners[j + 1], box_b_corners[j])

            cross_points[cnt] = ans_point

            if flag:
                poly_center = poly_center + cross_points[cnt]
                cnt += 1
    # check corners
    for k in range(4):
        if check_in_box2d(box_a, box_b_corners[k]):
            poly_center = poly_center + box_b_corners[k]
            cross_points[cnt] = box_b_corners[k]
            cnt += 1
        if check_in_box2d(box_b, box_a_corners[k]):
            poly_center = poly_center + box_a_corners[k]
            cross_points[cnt] = box_a_corners[k]
            cnt += 1

    if cnt != 0:
        poly_center.x /= cnt
        poly_center.y /= cnt
    # sort the points of polygon

    for j in range(cnt - 1):
        for i in range(cnt - j - 1):
            if point_cmp(cross_points[i], cross_points[i + 1], poly_center):
                temp = cross_points[i]
                cross_points[i] = cross_points[i + 1]
                cross_points[i + 1] = temp

    # get the overlap areas
    area = 0
    for k in range(cnt - 1):
        v1 = cross_points[k] - cross_points[0]
        v2 = cross_points[k + 1] - cross_points[0]
        area += cross(v1, v2, None)
    return fabs(area) / 2.0


def rotate_around_center(center: Point, angle_cos: float, angle_sin: float, p: Point) -> Point:
    new_x = (p.x - center.x) * angle_cos - (p.y - center.y) * angle_sin + center.x
    new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y
    p.set(new_x, new_y)
    return p


def check_in_box2d(box: List[float], p: Point):
    # params: box (7) [x, y, z, dx, dy, dz, heading]
    MARGIN = 1e-2

    center_x = box[0]
    center_y = box[1]
    # rotate the point in the opposite direction of box
    angle_cos = cos(-box[6])
    angle_sin = sin(-box[6])
    rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin)
    rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos

    return fabs(rot_x) < box[3] / 2 + MARGIN and fabs(rot_y) < box[4] / 2 + MARGIN


def point_cmp(a: Point, b: Point, center: Point):
    return atan2(a.y - center.y, a.x - center.x) > atan2(b.y - center.y, b.x - center.x)


def intersection(p1: Point, p0: Point, q1: Point, q0: Point):
    ans_point = Point()
    # fast exclusion
    if check_rect_cross(p0, p1, q0, q1) == 0:
        return 0, ans_point

    # check cross standing
    s1 = cross(q0, p1, p0)
    s2 = cross(p1, q1, p0)
    s3 = cross(p0, q1, q0)
    s4 = cross(q1, p1, q0)

    if not (s1 * s2 > 0 and s3 * s4 > 0):
        return 0, ans_point

    # calculate intersection of two lines
    s5 = cross(q1, p1, p0)
    if fabs(s5 - s1) > EPS:
        try:
            ans_point.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1)
            ans_point.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1)
        except ZeroDivisionError as e:
            print("intersection value can not be 0.")
    else:
        a0 = p0.y - p1.y
        b0 = p1.x - p0.x
        c0 = p0.x * p1.y - p1.x * p0.y
        a1 = q0.y - q1.y
        b1 = q1.x - q0.x
        c1 = q0.x * q1.y - q1.x * q0.y

        D = a0 * b1 - a1 * b0
        if D != 0:
            ans_point.x = (b0 * c1 - b1 * c0) / D
            ans_point.y = (a1 * c0 - a0 * c1) / D

    return 1, ans_point


def iou_bev(box_a: List[float], box_b: List[float]):
    # params box_a: [x, y, z, dx, dy, dz, heading]
    # params box_b: [x, y, z, dx, dy, dz, heading]
    sa = box_a[3] * box_a[4]
    sb = box_b[3] * box_b[4]
    s_overlap = box_overlap(box_a, box_b)
    max_val = max(sa + sb - s_overlap, EPS)
    try:
        result = s_overlap / max_val
    except ZeroDivisionError as e:
        print("value of area union can not be 0.")
    return result


class TestIoU3dNms3d:
    def cpu_to_exec(self, boxes, scores, threshold=0.0):
        boxes = boxes.numpy()
        order = scores.sort(0, descending=True)[1]
        order = order.numpy()
        boxes = boxes.take(order, 0)
        keep, num_out = self.cpu_nms_forward(boxes, threshold)
        keep = keep.astype(np.int64)
        keep = order[keep[:num_out]]
        return torch.from_numpy(keep)

    def cpu_nms_forward(self, boxes, nms_overlap_thresh=0.0):
        mask = np.ones(boxes.shape[0], dtype=int)
        keep = -np.ones(boxes.shape[0])
        num_out = 0
        for i in range(0, boxes.shape[0]):
            if mask[i] == 0:
                continue
            keep[num_out] = i
            num_out += 1
            for j in range(i + 1, boxes.shape[0]):
                if iou_bev(boxes[i], boxes[j]) > nms_overlap_thresh:
                    mask[j] = 0
        return keep, num_out
    
    def gen_input_data(self, boxes_num, dtype):
        boxes_xyz_coor = np.random.uniform(-1, 1, size=(boxes_num, 3)).astype(dtype)
        boxes_xyz_size_num = np.random.uniform(1, 50, size=(1, 3)).astype(dtype)
        boxes_xyz_size = (boxes_xyz_size_num * np.ones((boxes_num, 3))).astype(dtype)
        boxes_angle = np.radians(np.random.randint(0, 360, size=(boxes_num, 1))).astype(dtype)
        boxes = np.concatenate((boxes_xyz_coor, boxes_xyz_size), axis=1)
        boxes = np.concatenate((boxes, boxes_angle), axis = 1)
        boxes = torch.from_numpy(boxes)
        boxes_cuda = boxes.to("cuda")

        scores = np.random.uniform(1, boxes_num, size=(boxes_num)).astype(dtype)
        scores = torch.from_numpy(scores)
        scores_cuda = scores.to("cuda")

        return boxes, boxes_cuda, scores, scores_cuda
    
    def op_exec(self, boxes, scores, threshold):
        order = scores.sort(0, descending=True)[1]
        boxes = boxes[order].contiguous()

        keep = boxes.new_zeros(boxes.size(0), dtype=torch.long)
        num_out = boxes.new_zeros(size=(), dtype=torch.long)
        xav_dsal.iou3d_nms3d_forward(boxes, keep, num_out, nms_overlap_thresh=threshold)
        keep = order[keep[:num_out].to(boxes.device)].contiguous()
        return keep

    def one_case(self, boxes_num, dtype, threshold):
        boxes, boxes_cuda, scores, scores_cuda = self.gen_input_data(boxes_num, dtype)
        
        cpu_keep = self.op_exec(boxes, scores, threshold)
        xpu_keep = self.op_exec(boxes_cuda, scores_cuda, threshold).to("cpu")

        res_rel_error, res_abs_error = get_diff(dtype)
        ret = compare_tensors(xpu_keep, cpu_keep, res_rel_error, res_abs_error)
        assert ret
    
@pytest.mark.parametrize("boxes_num", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize("threshold", [0.7])
def test_iou3d_nms3d(boxes_num, dtype, threshold):
    IoU3d_nms3d = TestIoU3dNms3d()
    IoU3d_nms3d.one_case(boxes_num, dtype, threshold)


if __name__ == "__main__":
    pytest.main(["-v", "-s", "test_iou3d_nms3d.py"])