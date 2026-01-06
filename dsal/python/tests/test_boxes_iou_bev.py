import pytest
import torch
from torch import Tensor
import numpy as np
import xav_dsal
from data_compare import get_diff, compare_tensors


class TestBoxIoUBevForward:
    def gen_input_data(self, boxes_a_num, boxes_b_num, dtype):
        boxes_a_xyz_coor = np.random.uniform(-1, 1, size=(boxes_a_num, 3)).astype(dtype)
        boxes_a_xyz_size_num = np.random.uniform(1, 50, size=(1, 3)).astype(dtype)
        boxes_a_xyz_size = (boxes_a_xyz_size_num * np.ones((boxes_a_num, 3))).astype(dtype)
        boxes_a_angle = np.radians(np.random.randint(0, 360, size=(boxes_a_num, 1))).astype(dtype)
        boxes_a = np.concatenate((boxes_a_xyz_coor, boxes_a_xyz_size), axis=1)
        boxes_a = np.concatenate((boxes_a, boxes_a_angle), axis = 1)

        boxes_b_xyz_coor = np.random.uniform(-1, 1, size=(boxes_b_num, 3)).astype(dtype)
        boxes_b_xyz_size_num = np.random.uniform(1, 50, size=(1, 3)).astype(dtype)
        boxes_b_xyz_size = (boxes_b_xyz_size_num * np.ones((boxes_b_num, 3))).astype(dtype)
        boxes_b_angle = np.radians(np.random.randint(0, 360, size=(boxes_b_num, 1))).astype(dtype)
        boxes_b = np.concatenate((boxes_b_xyz_coor, boxes_b_xyz_size), axis=1)
        boxes_b = np.concatenate((boxes_b, boxes_b_angle), axis = 1)
        boxes_a = torch.from_numpy(boxes_a)
        boxes_b = torch.from_numpy(boxes_b)

        boxes_a_cuda = boxes_a.to("cuda")
        boxes_b_cuda = boxes_b.to("cuda")

        return boxes_a, boxes_b, boxes_a_cuda, boxes_b_cuda


    def one_case(self, boxes_a_num, boxes_b_num, dtype):
        boxes_a, boxes_b, boxes_a_cuda, boxes_b_cuda = self.gen_input_data(boxes_a_num, boxes_b_num, dtype)
        cpu_iou = xav_dsal.boxes_iou_bev(boxes_a, boxes_b)
        cpu_iou = cpu_iou.to("cpu")

        rel_iou = xav_dsal.boxes_iou_bev(boxes_a_cuda, boxes_b_cuda)
        rel_iou = rel_iou.to("cpu")

        res_rel_error, res_abs_error = get_diff(dtype)
        ret = compare_tensors(cpu_iou, rel_iou, res_rel_error, 3e-4)
        assert ret

@pytest.mark.parametrize("boxes_a_num", [16, 32, 128])
@pytest.mark.parametrize("boxes_b_num", [16, 27, 56])
@pytest.mark.parametrize("dtype", [np.float32])
def test_boxes_iou_bev(boxes_a_num, boxes_b_num, dtype):
    boxes_iou_bev_fwd = TestBoxIoUBevForward()
    boxes_iou_bev_fwd.one_case(boxes_a_num, boxes_b_num, dtype)


if __name__ == "__main__":
    pytest.main(["-v", "-s", "test_boxes_iou_bev.py"])