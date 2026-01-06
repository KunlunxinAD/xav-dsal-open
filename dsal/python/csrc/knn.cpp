#include "xpytorch.hpp"
/*
Args:
            k (int): number of nearest neighbors.
            xyz (torch.Tensor): (B, N, 3) if transposed == False, else
                (B, 3, N). xyz coordinates of the features.
            center_xyz (torch.Tensor, optional): (B, npoint, 3) if transposed
                is False, else (B, 3, npoint). centers of the knn query.
                Default: None.
            transposed (bool, optional): whether the input tensors are
                transposed. Should not explicitly use this keyword when
                calling knn (=KNN.apply), just add the fourth param.
                Default: False.
*/

std::vector<at::Tensor> knn(int64_t k_, at::Tensor xyz, c10::optional<at::Tensor> center_xyz, bool transposed = false) {
    int k = static_cast<int>(k_);
    auto ctx = xmlir_rt::getXpuKernelContext();
    AT_ASSERTM((k > 0) & (k < 100), "k should be in range(0, 100)");
    TORCH_CHECK(xyz.is_contiguous(), " must be contiguous");
    AT_ASSERTM(xyz.dim() == 3, "xyz must be 3 dim");

    at::Tensor center_xyz_value;

    if (!center_xyz.has_value()) {
        center_xyz_value = xyz;
    } else {
        AT_ASSERTM(xyz.size(0) == center_xyz->size(0), "must be same batch size");
        TORCH_CHECK(center_xyz->is_contiguous(), " must be contiguous");
        AT_ASSERTM(center_xyz->dim() == 3, "xyz must be 3 dim");
        center_xyz_value = center_xyz->to(at::kFloat);
    }

    at::Tensor xyz_trans;
    if (transposed) {
        xyz_trans = xyz;
        xyz = xyz.transpose(2, 1).contiguous();
        center_xyz_value = center_xyz_value.transpose(2, 1).contiguous();
    } else {
        xyz_trans = xyz.transpose(2, 1).contiguous();
    }
    AT_ASSERTM(xyz.size(2) == 3, "xyz must be 3 channel");
    AT_ASSERTM(center_xyz_value.size(2) == 3, "center_xyz must be 3 channel");

    int b = xyz.size(0);
    int n = xyz.size(1);
    int m = center_xyz_value.size(1);

    auto output_idx = at::full({b, m, k}, -1, at::TensorOptions().dtype(at::kInt).device(xyz.device()));
    auto output_dist = at::full({b, m, k}, -1.0f, at::TensorOptions().dtype(at::kFloat).device(xyz.device()));

    int ret = 0;

    if (xyz.device().is_cuda()) {
        ret = xav::xpu::knn<float>(
                ctx,
                b,
                n,
                m,
                k,
                xyz_trans.data_ptr<float>(),
                center_xyz_value.data_ptr<float>(),
                output_idx.data_ptr<int>(),
                output_dist.data_ptr<float>());
    } else {
        ret = xav::cpu::knn<float>(
                ctx,
                b,
                n,
                m,
                k,
                xyz.data_ptr<float>(),
                center_xyz_value.data_ptr<float>(),
                output_idx.data_ptr<int>(),
                output_dist.data_ptr<float>());
    }
    assert(ret == 0);
    // idx shape to [B, k, m]
    output_idx = output_idx.transpose(2, 1).contiguous();
    return {output_idx, output_dist};
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("knn", &knn);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("knn", &knn);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("knn(int k, Tensor xyz, Tensor? center_xyz, bool transposed=False) -> Tensor[]"));
}