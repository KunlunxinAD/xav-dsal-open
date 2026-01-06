import pytest
import torch
import os
import pandas as pd
import numpy as np
from data_compare import compare_tensors

import xav_dsal


@pytest.fixture
def base_path():
    return "./data/test_smooth_cosine_loss/"


def read_from_file(base_path):
    emb_gt_file = f"{base_path}emb_gt.pt"
    offset_feat_file = f"{base_path}offset_feat.pt"
    h_pix_file = f"{base_path}h_pix.pt"
    w_pix_file = f"{base_path}w_pix.pt"
    ignore_label_file = f"{base_path}ignore_label.pt"

    required_files = [emb_gt_file, offset_feat_file, h_pix_file, w_pix_file, ignore_label_file]

    for file_path in required_files:
        assert os.path.exists(file_path), f"Test file does not exist: {file_path}"

    emb_gt = torch.load(emb_gt_file, weights_only=True)
    offset_feat = torch.load(offset_feat_file, weights_only=True)
    h_pix = torch.load(h_pix_file, weights_only=True)
    w_pix = torch.load(w_pix_file, weights_only=True)
    ignore_label = torch.load(ignore_label_file, weights_only=True)

    return (emb_gt, offset_feat, h_pix, w_pix, ignore_label)


def generate_fake_emb_gt(
    batch_size,
    bev_h,
    bev_w,
    ignore_label,
    instance_coverage=0.7,  # 实例覆盖的比例（0~1，越高背景越少）
    min_region_size=5,  # 实例区域最小尺寸（如3x3）
    max_region_size=7,  # 实例区域最大尺寸（如7x7）
    noise_ratio=0.05,
):
    """
    生成少背景（instance=0）的float32类型torch假数据

    参数调整：
        instance_coverage: 控制实例覆盖的区域比例（默认0.7，即70%区域为实例）
        min_region_size/max_region_size: 增大实例区域尺寸，减少背景
    """
    # 1. 初始化全0数组（背景）
    emb_gt_np = np.zeros((batch_size, bev_h, bev_w), dtype=np.int32)
    max_instance_id = ignore_label - 1
    if max_instance_id < 1:
        raise ValueError("ignore_label必须大于1")

    total_pixels = bev_h * bev_w  # 单batch的总像素数
    target_pixels = int(total_pixels * instance_coverage)  # 目标覆盖的实例像素数

    for b in range(batch_size):
        covered = 0  # 已覆盖的实例像素数
        inst_id = 1  # 实例ID计数器

        # 循环生成实例，直到覆盖足够的像素
        while covered < target_pixels:
            # 实例ID循环（不超过max_instance_id）
            current_id = inst_id % max_instance_id
            if current_id == 0:
                current_id = max_instance_id
            inst_id += 1

            # 随机种子点（尽量均匀分布）
            seed_h = np.random.randint(0, bev_h)
            seed_w = np.random.randint(0, bev_w)

            # 随机区域大小（比之前更大，减少背景）
            region_size = np.random.randint(min_region_size, max_region_size + 1)
            half = region_size // 2

            # 生成实例区域
            for dh in range(-half, half + 1):
                for dw in range(-half, half + 1):
                    h = seed_h + dh
                    w = seed_w + dw
                    if 0 <= h < bev_h and 0 <= w < bev_w:
                        # 已覆盖的点不再重复计算
                        if emb_gt_np[b, h, w] == 0:
                            covered += 1
                        # 噪声控制（少数点可能被覆盖为其他实例）
                        if np.random.rand() > noise_ratio:
                            emb_gt_np[b, h, w] = current_id

            # 防止无限循环（极端情况退出）
            if inst_id > max_instance_id * 10:
                break

    # 转换为float32的torch张量
    emb_gt_torch = torch.tensor(emb_gt_np, dtype=torch.float32)
    return emb_gt_torch


def smooth_cosine_loss_forward(emb_gt, offset_feat, h_pix, w_pix, ignore_label):
    emb_gt_cpu = emb_gt.to("cpu").detach()
    offset_feat_cpu = offset_feat.to("cpu").detach()
    output_cpu = xav_dsal.smooth_cosine_loss_forward(emb_gt_cpu, offset_feat_cpu, h_pix, w_pix, ignore_label)
    print("output_cpu", output_cpu)

    emb_gt_cuda = emb_gt.to("cuda").detach()
    offset_feat_cuda = offset_feat.to("cuda").detach()
    output_cuda = xav_dsal.smooth_cosine_loss_forward(emb_gt_cuda, offset_feat_cuda, h_pix, w_pix, ignore_label)
    output_cuda = output_cuda.to("cpu")
    print("output_cuda", output_cuda)

    return compare_tensors(output_cpu, output_cuda, 1e-4, 1e-4, "aten.smooth_cosine_loss_forward.default")


def smooth_cosine_loss_backward(emb_gt, offset_feat, grad_offset_feat, grad_output, h_pix, w_pix, ignore_label):
    print("emb_gt size", emb_gt.size())
    print("offset_feat size", offset_feat.size())
    print("grad_offset_feat size", grad_offset_feat.size())
    print("grad_output size", grad_output.size())
    print("h_pix", h_pix)
    print("w_pix", w_pix)
    print("ignore_label", ignore_label)

    emb_gt_cpu = emb_gt.to("cpu").detach()
    offset_feat_cpu = offset_feat.to("cpu").detach()
    grad_offset_feat_cpu = grad_offset_feat.to("cpu").detach()
    grad_output = grad_output.to("cpu").detach()
    xav_dsal.smooth_cosine_loss_backward(
        emb_gt_cpu, offset_feat_cpu, grad_offset_feat_cpu, grad_output, h_pix, w_pix, ignore_label
    )

    emb_gt_cuda = emb_gt.to("cuda").detach()
    offset_feat_cuda = offset_feat.to("cuda").detach()
    grad_offset_feat_cuda = grad_offset_feat.to("cuda").detach()
    grad_output = grad_output.to("cuda").detach()
    xav_dsal.smooth_cosine_loss_backward(
        emb_gt_cuda, offset_feat_cuda, grad_offset_feat_cuda, grad_output, h_pix, w_pix, ignore_label
    )
    grad_offset_feat_cuda = grad_offset_feat_cuda.to("cpu")

    # print(grad_offset_feat_cpu)
    # print(grad_offset_feat_cuda)

    return compare_tensors(
        grad_offset_feat_cpu, grad_offset_feat_cuda, 1e-3, 1e-3, "aten.smooth_cosine_loss_backward.default"
    )


@pytest.mark.parametrize("test_id", range(5))
def test_smooth_cosloss_forward(base_path, test_id):
    read_file = False
    read_pt = True
    save_pt = False
    if read_file:
        (emb_gt, offset_feat, h_pix, w_pix, ignore_label) = read_from_file(base_path)
        print(emb_gt.size())
        print(offset_feat.size())
        print(h_pix)
        print(w_pix)
        emb_gt = generate_fake_emb_gt(emb_gt.size()[0], emb_gt.size()[1], emb_gt.size()[2], ignore_label).to(
            torch.float32
        )
        assert smooth_cosine_loss_forward(emb_gt, offset_feat, h_pix, w_pix, ignore_label)
        return

    # fmt: off
    batch_size      = [1,   8,   14,  10,  15]
    bev_h           = [12,  200, 150, 250, 200]
    bev_w           = [10,  160, 200, 340, 350]
    h_pix           = [0.8, 0.8, 0.9, 0.6, 0.7]
    w_pix           = [0.8, 0.8, 0.9, 0.8, 0.9]
    ignore_label    = [100, 355, 287, 304, 253]
    # fmt: on

    if read_pt:
        emb_gt = torch.load(f"data/test_smooth_cosine_loss/forward/emb_gt_{test_id}.pt", weights_only=True)
        offset_feat = torch.load(f"data/test_smooth_cosine_loss/forward/offset_feat_{test_id}.pt", weights_only=True)
    else:
        emb_gt = generate_fake_emb_gt(batch_size[test_id], bev_h[test_id], bev_w[test_id], ignore_label[test_id]).to(
            torch.float32
        )
        offset_feat = torch.randn((batch_size[test_id], bev_h[test_id], bev_w[test_id], 3), dtype=torch.float32)
        if save_pt:
            torch.save(emb_gt, f"data/test_smooth_cosine_loss/forward/emb_gt_{test_id}.pt")
            torch.save(offset_feat, f"data/test_smooth_cosine_loss/forward/offset_feat_{test_id}.pt")
    res = smooth_cosine_loss_forward(emb_gt, offset_feat, h_pix[test_id], w_pix[test_id], ignore_label[test_id])
    assert res


@pytest.mark.parametrize("test_id", range(8))
def test_smooth_cosloss_backward(test_id):
    # fmt: off
    batch_size =    [1,   8,   10,  12,  15,  2,   8,   14]
    bev_h =         [20,  200, 150, 250, 200, 66,  200, 120]
    bev_w =         [20,  160, 200, 290, 160, 64,  360, 370]
    h_pix =         [0.8, 0.8, 0.9, 0.6, 0.7, 0.8, 0.8, 0.7]
    w_pix =         [0.8, 0.8, 0.9, 0.8, 0.9, 0.8, 0.8, 0.8]
    ignore_label =  [100, 190, 287, 255, 253, 250, 216, 208]
    # fmt: on
    read_pt = True
    save_pt = False

    if read_pt:
        emb_gt = torch.load(f"data/test_smooth_cosine_loss/backward/emb_gt_{test_id}.pt", weights_only=True)
        offset_feat = torch.load(f"data/test_smooth_cosine_loss/backward/offset_feat_{test_id}.pt", weights_only=True)
        grad_output = torch.load(f"data/test_smooth_cosine_loss/backward/grad_output_{test_id}.pt", weights_only=True)
    else:
        emb_gt = generate_fake_emb_gt(batch_size[test_id], bev_h[test_id], bev_w[test_id], ignore_label[test_id]).to(
            torch.float32
        )
        offset_feat = torch.randn((batch_size[test_id], bev_h[test_id], bev_w[test_id], 3), dtype=torch.float32)
        grad_output = torch.randn((1), dtype=torch.float32)
        if save_pt:
            torch.save(grad_output, f"data/test_smooth_cosine_loss/backward/grad_output_{test_id}.pt")
            torch.save(emb_gt, f"data/test_smooth_cosine_loss/backward/emb_gt_{test_id}.pt")
            torch.save(offset_feat, f"data/test_smooth_cosine_loss/backward/offset_feat_{test_id}.pt")

    grad_offset_feat = torch.empty((batch_size[test_id], bev_h[test_id], bev_w[test_id], 3), dtype=torch.float32)
    res = smooth_cosine_loss_backward(
        emb_gt, offset_feat, grad_offset_feat, grad_output, h_pix[test_id], w_pix[test_id], ignore_label[test_id]
    )
    assert res


if __name__ == "__main__":
    pytest.main(["-v", "-s", "test_smooth_cosloss.py"])
