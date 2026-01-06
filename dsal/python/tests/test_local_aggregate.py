import pytest
import torch
import xav_dsal
from data_compare import compare_int_tensors, compare_tensors


def generate_random_data(P=500, H=16, W=16, D=4, N=1000):
    pts = torch.rand(N, 3, dtype=torch.float32)  # [0,1) 随机
    pts[:, 0] *= H  # x 坐标范围 [0, H)
    pts[:, 1] *= W  # y 坐标范围 [0, W)
    pts[:, 2] *= D  # z 坐标范围 [0, D)

    # 2. 生成 points_int: (N, 3) 整数型点坐标（pts 的量化结果）
    # 通常是 pts 四舍五入或取整，与 pts 语义对应
    points_int = torch.round(pts).to(torch.int32)
    points_int[:, 0] = torch.clamp(points_int[:, 0], 0, H - 1)
    points_int[:, 1] = torch.clamp(points_int[:, 1], 0, W - 1)
    points_int[:, 2] = torch.clamp(points_int[:, 2], 0, D - 1)

    # 3. 生成 means3D: (P, 3) 浮点型 Gaussian 中心坐标
    # 与 pts 关联：可从 pts 中采样或添加噪声（模拟 Gaussian 围绕点分布）
    if P <= N:
        # 从 pts 中随机采样 P 个点作为中心（加微小噪声）
        indices = torch.randint(0, N, (P,))
        means3D = pts[indices] + torch.randn(P, 3) * 0.1  # 加噪声
    else:
        # 若 P > n，复制 pts 并添加噪声
        means3D = pts.repeat(P // N + 1, 1)[:P] + torch.randn(P, 3) * 0.1
    # 确保中心坐标在网格范围内
    means3D[:, 0] = torch.clamp(means3D[:, 0], 0, H - 1)
    means3D[:, 1] = torch.clamp(means3D[:, 1], 0, W - 1)
    means3D[:, 2] = torch.clamp(means3D[:, 2], 0, D - 1)

    # 4. 生成 means3D_int: (P, 3) 整数型中心坐标（means3D 的量化）
    means3D_int = torch.round(means3D).to(torch.int32)
    # 确保整数坐标在网格范围内（避免 tile 索引越界）
    means3D_int[:, 0] = torch.clamp(means3D_int[:, 0], 0, H - 1)
    means3D_int[:, 1] = torch.clamp(means3D_int[:, 1], 0, W - 1)
    means3D_int[:, 2] = torch.clamp(means3D_int[:, 2], 0, D - 1)

    # 5. 生成 opacities: (P,) 不透明度（0-1 之间）
    opacities = torch.rand(P, dtype=torch.float32)  # [0,1) 随机

    # 6. 生成 semantics: (P, c) 语义特征（c 为语义通道数，这里设为 3）
    c = 18  # 示例通道数，可根据实际需求调整
    semantics = torch.clamp(torch.randn(P, c, dtype=torch.float32) + 5, 0, 1)  # 随机语义特征

    # 7. 生成 radii: (P,) 整数半径（≥1，控制 Gaussian 覆盖的 tile 范围）
    radii = torch.randint(1, 4, (P,), dtype=torch.int32)  # 半径 1-3（避免过大）

    # # 8. 生成 cov3D: (P, 6) 协方差矩阵（3x3 对称矩阵的 6 个独立元素）
    # # 3x3 对称矩阵格式：[a, b, c, d, e, f] 对应 [[a,b,c],[b,d,e],[c,e,f]]
    # # 确保正定（对角线元素为正，避免奇异）
    cov3D = torch.randn(P, 6, dtype=torch.float32) * 0.1  # 小范围随机
    cov3D[:, 0] += 1.0  # 对角线 a = 1.0 + 噪声（保证正定）
    cov3D[:, 3] += 1.0  # 对角线 d = 1.0 + 噪声
    cov3D[:, 5] += 1.0  # 对角线 f = 1.0 + 噪声

    return {
        "pts": pts,
        "points_int": points_int,
        "means3D": means3D,
        "means3D_int": means3D_int,
        "opacities": opacities,
        "semantics": semantics,
        "radii": radii,
        "cov3D": cov3D,
        "H": H,
        "W": W,
        "D": D,
    }


def generate_out_grad(N, C, dtype=torch.float32, dist="normal"):
    if dist == "normal":
        # 正态分布：mean=0, std=0.1（可根据需求调整 std，如 0.5）
        out_grad = torch.randn((N, C), dtype=dtype) * 0.1
    elif dist == "uniform":
        # 均匀分布：范围 [-0.5, 0.5]（可调整为 [-1, 1]）
        out_grad = torch.rand((N, C), dtype=dtype) - 0.5
    else:
        raise ValueError("dist 仅支持 'normal' 或 'uniform'")
    return out_grad


def compare_tensor_results(result_cpu, result_cuda, type):
    if type == "fwd":
        assert compare_int_tensors(result_cpu[0], result_cuda[0], "point_offsets")
        assert compare_int_tensors(result_cpu[1], result_cuda[1], "point_list_keys_unsorted")
        assert compare_int_tensors(result_cpu[2], result_cuda[2], "point_list_unsorted")
        assert compare_int_tensors(result_cpu[3], result_cuda[3], "ranges")

        temp = torch.abs(result_cpu[4])
        mul = 1
        if temp.mean() < 0.5:
            mul = int(1 / temp.mean())
        assert compare_tensors(result_cpu[4] * mul, result_cuda[4] * mul, 1e-4, 1e-4, "out_logits")
    elif type == "bwd":
        names = ("means3D_grad", "opacity_grad", "semantics_grad", "cov3D_grad")
        for i in range(4):
            mul = 1
            temp = torch.abs(result_cpu[1])
            if temp.mean() < 0.5:
                mul = int(1 / temp.mean())
            assert compare_tensors(result_cpu[i] * mul, result_cuda[i] * mul, 1e-4, 1e-4, names[i])
        # assert compare_tensors(result_cpu[0], result_cuda[0], 1e-4, 1e-4, "means3D_grad")
        # assert compare_tensors(result_cpu[1], result_cuda[1], 1e-4, 1e-4, "opacity_grad")
        # assert compare_tensors(result_cpu[2], result_cuda[2], 1e-4, 1e-4, "semantics_grad")
        # assert compare_tensors(result_cpu[3], result_cuda[3], 1e-4, 1e-4, "cov3D_grad")
        assert compare_int_tensors(result_cpu[4], result_cuda[4], "voxel2pts")


def run_fwd(pts, points_int, means3D, means3D_int, opacities, semantics, radii, cov3D, H, W, D, device):
    (num_rendered, point_offsets, point_list_keys_unsorted, point_list_unsorted, ranges, out_logits) = (
        xav_dsal.local_aggregate_forward(
            pts.to(device),
            points_int.to(device),
            means3D.to(device),
            means3D_int.to(device),
            opacities.to(device),
            semantics.to(device),
            radii.to(device),
            cov3D.to(device),
            H,
            W,
            D,
        )
    )
    return (
        num_rendered,
        point_offsets.to("cpu"),
        point_list_keys_unsorted.to("cpu"),
        point_list_unsorted.to("cpu"),
        ranges.to("cpu"),
        out_logits.to("cpu"),
    )


def run_bwd(
    point_offsets,
    point_list_keys_unsorted,
    means3D,
    pts,
    points_int,
    cov3D,
    opacities,
    semantics,
    out_grad,
    H,
    W,
    D,
    R,
    device,
):
    (means3D_grad, opacity_grad, semantics_grad, cov3D_grad, voxel2pts) = xav_dsal.local_aggregate_backward(
        point_offsets.to(device),
        point_list_keys_unsorted.to(device),
        means3D.to(device),
        pts.to(device),
        points_int.to(device),
        cov3D.to(device),
        opacities.to(device),
        semantics.to(device),
        out_grad.to(device),
        H,
        W,
        D,
        R,
    )
    return (
        means3D_grad.to("cpu"),
        opacity_grad.to("cpu"),
        semantics_grad.to("cpu"),
        cov3D_grad.to("cpu"),
        voxel2pts.to("cpu"),
    )


@pytest.fixture
def base_path():
    return "./data/test_local_aggregate/"


@pytest.mark.parametrize("test_id", [0, 1, 2, 3])
def test_local_aggregate_fwd(base_path, test_id):
    read_pt = True
    save_pt = False
    compare_file = False  # can only compare test_id 0 result
    # test_id = 0

    P = [25601, 500, 100, 10000]
    H = [200, 16, 8, 50]
    W = [200, 16, 8, 50]
    D = [16, 4, 2, 16]
    N = [640000, 1000, 200, 30000]

    if read_pt:
        data = torch.load(f"{base_path}local_aggregate_fwd_{test_id}.pt", weights_only=True)
    else:
        data = generate_random_data(P[test_id], H[test_id], W[test_id], D[test_id], N[test_id])
        if save_pt:
            torch.save(data, f"{base_path}local_aggregate_fwd_{test_id}.pt")

    for key, value in data.items():
        if type(value) == torch.Tensor:
            print(key, value.shape, value.dtype, value.device)
        else:
            print(key, value)

    pts = data["pts"]
    points_int = data["points_int"]
    means3D = data["means3D"]
    means3D_int = data["means3D_int"]
    semantics = data["semantics"]
    opacities = data["opacities"]
    radii = data["radii"]
    cov3D = data["cov3D"]
    H = data["H"]
    W = data["W"]
    D = data["D"]

    if compare_file:
        out_data = torch.load(f"{base_path}local_aggregate_fwd_output_0.pt", weights_only=True)
        # out_tiles_touched = out_data["tiles_touched"]
        out_points_offset = out_data["points_offset"]
        out_point_list_keys_unsorted = out_data["point_list_keys_unsorted"]
        out_point_list_unsorted = out_data["point_list_unsorted"]
        # out_point_list_keys = out_data["point_list_keys"]
        # out_point_list = out_data["point_list"]
        out_ranges = out_data["ranges"]
        out_logits = out_data["out_logits"]

        result_cpu = (
            out_data["R"],
            # out_tiles_touched,
            out_points_offset,
            out_point_list_keys_unsorted,
            out_point_list_unsorted,
            # out_point_list_keys,
            out_ranges,
            out_logits,
        )
    else:
        result_cpu = run_fwd(
            pts,
            points_int,
            means3D,
            means3D_int,
            opacities,
            semantics,
            radii,
            cov3D,
            H,
            W,
            D,
            "cpu",
        )
        print("num_rendered_cpu", result_cpu[0])

    result_cuda = run_fwd(
        pts,
        points_int,
        means3D,
        means3D_int,
        opacities,
        semantics,
        radii,
        cov3D,
        H,
        W,
        D,
        "cuda",
    )
    print("num_rendered_cuda", result_cuda[0])

    assert result_cpu[0] == result_cuda[0]  # num_rendered
    compare_tensor_results(result_cpu[1:], result_cuda[1:], "fwd")


def ttest_local_aggregate_bwd(base_path):
    derive_from_fwd = False
    compare_file = True
    test_id = 0

    P = [25601, 500, 100, 10000]
    H = [200, 16, 8, 50]
    W = [200, 16, 8, 50]
    D = [16, 4, 2, 16]
    N = [640000, 1000, 200, 30000]

    if derive_from_fwd:
        fwd_intput_data = torch.load(f"{base_path}local_aggregate_fwd_{test_id}.pt", weights_only=True)
        fwd_out_data = torch.load(f"{base_path}local_aggregate_fwd_output_{test_id}.pt", weights_only=True)

        for key, value in fwd_intput_data.items():
            if type(value) == torch.Tensor:
                print(key, value.shape, value.dtype, value.device)
            else:
                print(key, value)

        out_grad = generate_out_grad(N[test_id], 18)
        pts = fwd_intput_data["pts"]
        points_int = fwd_intput_data["points_int"]
        means3D = fwd_intput_data["means3D"]
        semantics = fwd_intput_data["semantics"]
        opacities = fwd_intput_data["opacities"]
        cov3D = fwd_intput_data["cov3D"]
        H = fwd_intput_data["H"]
        W = fwd_intput_data["W"]
        D = fwd_intput_data["D"]

        R = fwd_out_data["R"]
        out_points_offset = fwd_out_data["points_offset"]
        point_list_keys_unsorted = fwd_out_data["point_list_keys_unsorted"]

        # bwd_out_data = torch.load(f"{base_path}local_aggregate_bwd_output_{test_id}.pt", weights_only=True)
        # result_cpu = (
        #     bwd_out_data["means3D_grad"].to("cpu"),
        #     bwd_out_data["opacity_grad"].to("cpu"),
        #     bwd_out_data["semantics_grad"].to("cpu"),
        #     bwd_out_data["cov3D_grad"].to("cpu"),
        # )

        result_cpu = run_bwd(
            out_points_offset,
            point_list_keys_unsorted,
            means3D,
            pts,
            points_int,
            cov3D,
            opacities,
            semantics,
            out_grad,
            H,
            W,
            D,
            R,
            "cpu",
        )
    elif compare_file:
        archive_input = torch.jit.load(f"{base_path}/local_aggregate_bwd.pt")
        archive_output = torch.jit.load(f"{base_path}/local_aggregate_bwd_output.pt")

        out_points_offset = archive_input.points_offset_tensor
        point_list_keys_unsorted = archive_input.point_list_keys_unsorted_tensor
        pts = archive_input.pts_cpu_tensor
        points_int = archive_input.points_int_tensor
        means3D = archive_input.means3D_tensor
        cov3D = archive_input.cov3D_tensor
        opacities = archive_input.opacities_tensor
        semantics = archive_input.semantics_tensor
        out_grad = archive_input.out_grad_tensor
        H = archive_input.H_tensor
        W = archive_input.W_tensor
        D = archive_input.D_tensor
        R = archive_input.R_tensor

        result_cpu = (
            archive_output.means3d_grad_tensor,
            archive_output.opacity_grad_tensor,
            archive_output.semantics_grad_tensor,
            archive_output.cov3D_grad_tensor,
            archive_output.voxel2pts_tensor,
        )

    result_cuda = run_bwd(
        out_points_offset,
        point_list_keys_unsorted,
        means3D,
        pts,
        points_int,
        cov3D,
        opacities,
        semantics,
        out_grad,
        H,
        W,
        D,
        R,
        "cuda",
    )

    compare_tensor_results(result_cpu, result_cuda, "bwd")


if __name__ == "__main__":
    pytest.main(["-v", "-s", "test_local_aggregate.py"])
