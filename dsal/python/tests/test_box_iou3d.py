import pytest
import torch
from torch import Tensor
import numpy as np
import xav_dsal
from data_compare import get_diff, compare_tensors


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
    xav_dsal.iou3d_boxes_overlap_bev_forward(boxes_a.contiguous(),
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


class TestBoxIoU3DForward:
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
        cpu_iou = boxes_iou3d(boxes_a, boxes_b)
        cpu_iou = cpu_iou.to("cpu")

        rel_iou = boxes_iou3d(boxes_a_cuda, boxes_b_cuda)
        rel_iou = rel_iou.to("cpu")

        res_rel_error, res_abs_error = get_diff(dtype)
        ret = compare_tensors(rel_iou, cpu_iou, res_rel_error, 3e-4)
        assert ret

@pytest.mark.parametrize("boxes_a_num", [128, 256, 512, 2048])
@pytest.mark.parametrize("boxes_b_num", [128, 256, 512, 1024])
@pytest.mark.parametrize("dtype", [np.float32])
def test_boxes_iou3d(boxes_a_num, boxes_b_num, dtype):
    boxes_iou3d_fwd = TestBoxIoU3DForward()
    boxes_iou3d_fwd.one_case(boxes_a_num, boxes_b_num, dtype)


if __name__ == "__main__":
    pytest.main(["-v", "-s", "test_box_iou3d.py"])
