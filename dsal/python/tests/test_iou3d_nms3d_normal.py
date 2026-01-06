import pytest
import torch
import os
import numpy as np
import xav_dsal
from typing import List
from data_compare import get_diff, compare_tensors
from data_cache import golden_data_cache, load_data

EPS = 1e-8

def iou_normal(a: List[float], b: List[float]):
    left = max(a[0] - a[3] / 2.0, b[0] - b[3] / 2.0)
    right = min(a[0] + a[3] / 2.0, b[0] + b[3] / 2.0)
    top = max(a[1] - a[4] / 2.0, b[1] - b[4] / 2.0)
    bottom = min(a[1] + a[4] / 2.0, b[1] + b[4] / 2.0)
    width = max(right - left, 0.0)
    height = max(bottom - top, 0.0)
    interS = width * height
    Sa = a[3] * a[4]
    Sb = b[3] * b[4]
    return interS / max(Sa + Sb - interS, EPS)

class TestIoU3dNms3dNormal:
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
                if iou_normal(boxes[i], boxes[j]) > nms_overlap_thresh:
                    mask[j] = 0
        return keep, num_out
    
    # @golden_data_cache(__file__, refresh_data=True)
    def gen_input_data(self, boxes_num, dtype):
        boxes_xyz_coor = np.random.uniform(-1, 1, size=(boxes_num, 3)).astype(dtype)
        boxes_xyz_size_num = np.random.uniform(1, 50, size=(1, 3)).astype(dtype)
        boxes_xyz_size = (boxes_xyz_size_num * np.ones((boxes_num, 3))).astype(dtype)
        boxes_angle = np.radians(np.random.randint(0, 360, size=(boxes_num, 1))).astype(dtype)
        boxes = np.concatenate((boxes_xyz_coor, boxes_xyz_size), axis=1)
        boxes = np.concatenate((boxes, boxes_angle), axis = 1)
        boxes = torch.from_numpy(boxes)

        scores = np.random.uniform(1, boxes_num, size=(boxes_num)).astype(dtype)
        scores = torch.from_numpy(scores)

        return boxes, scores
    
    def op_exec(self, boxes, scores, threshold):
        order = scores.sort(0, descending=True)[1]
        boxes = boxes[order].contiguous()

        keep = boxes.new_zeros(boxes.size(0), dtype=torch.long)
        num_out = boxes.new_zeros(size=(), dtype=torch.long)
        xav_dsal.iou3d_nms3d_normal_forward(boxes, keep, num_out, nms_overlap_thresh=threshold)
        keep = order[keep[:num_out].to(boxes.device)].contiguous()
        return keep

    def one_case(self, boxes_num, dtype, threshold):
        boxes, scores = self.gen_input_data(boxes_num, dtype)
        # save_path = os.path.dirname(os.path.abspath(__file__)) + "/data_cache/test_iou3d_nms3d_normal/gen_input_data/"
        # file_names = ["e24b14df20_494b540c9e_798e908e9a_0.pth", "e24b14df20_494b540c9e_798e908e9a_1.pth"]
        # boxes, scores = load_data(save_path, file_names)

        boxes_cuda = boxes.to("cuda")
        scores_cuda = scores.to("cuda")
        
        python_keep = self.cpu_to_exec(boxes, scores, threshold)
        xpu_keep = self.op_exec(boxes_cuda, scores_cuda, threshold).to("cpu")

        res_rel_error, res_abs_error = get_diff(dtype)
        ret = compare_tensors(xpu_keep, python_keep, res_rel_error, res_abs_error)
        assert ret
    
@pytest.mark.parametrize("boxes_num", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize("threshold", [0.7])
def test_iou3d_nms3d(boxes_num, dtype, threshold):
    IoU3d_nms3d = TestIoU3dNms3dNormal()
    IoU3d_nms3d.one_case(boxes_num, dtype, threshold)


if __name__ == "__main__":
    pytest.main(["-v", "-s", "test_iou3d_nms3d_normal.py"])