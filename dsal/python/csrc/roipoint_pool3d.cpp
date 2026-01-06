#include "xpytorch.hpp"

void roipoint_pool3d_forward(const at::Tensor& points, const at::Tensor& boxes3d, const at::Tensor& point_features, 
                            at::Tensor& pooled_features, at::Tensor& pooled_empty_flag) {
    auto ctx = xmlir_rt::getXpuKernelContext();

    TORCH_CHECK(points.is_contiguous(), " must be contiguous");
    TORCH_CHECK(boxes3d.is_contiguous(), " must be contiguous");
    TORCH_CHECK(point_features.is_contiguous(), " must be contiguous");
    TORCH_CHECK(pooled_features.is_contiguous(), " must be contiguous");
    TORCH_CHECK(pooled_empty_flag.is_contiguous(), " must be contiguous");

    int batch_size = points.size(0);
    int pts_num = points.size(1);
    int boxes_num = boxes3d.size(1);
    int feature_in_len = point_features.size(2);
    int sampled_pts_num = pooled_features.size(2);

    AT_ASSERTM(sampled_pts_num <= 512, "sampled_pts_num must <= 512");
    int ret = 0;
    if (boxes3d.device().is_cuda()) {
        ret = xav::xpu::roipoint_pool3d_forward<float>(
            ctx,
            batch_size,
            pts_num,
            boxes_num,
            feature_in_len,
            sampled_pts_num,
            points.data_ptr<float>(), // (3, B, pts_num)
            point_features.data_ptr<float>(), // (B, pts_num, feature_in_len)
            boxes3d.data_ptr<float>(), // (B, boxes_num, 7)
            pooled_features.data_ptr<float>(), // (B, boxes_num, sampled_pts_num, feature_in_len + 3)
            pooled_empty_flag.data_ptr<int>());
    } else {
        printf("Enter cpu version\n");
        ret = xav::cpu::roipoint_pool3d_forward<float>(
            ctx,
            batch_size,
            pts_num,
            boxes_num,
            feature_in_len,
            sampled_pts_num,
            points.data_ptr<float>(), // (3, B, pts_num)
            point_features.data_ptr<float>(), // (B, pts_num, feature_in_len)
            boxes3d.data_ptr<float>(), // (B, boxes_num, 7)
            pooled_features.data_ptr<float>(), // (B, boxes_num, sampled_pts_num, feature_in_len + 3)
            pooled_empty_flag.data_ptr<int>());
    }
    assert(ret == 0);
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("roipoint_pool3d_forward", &roipoint_pool3d_forward);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("roipoint_pool3d_forward", &roipoint_pool3d_forward);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("roipoint_pool3d_forward(Tensor points, Tensor boxes3d, Tensor point_features, "
        "Tensor(a!) pooled_features, Tensor(b!) pooled_empty_flag) -> ()"));
}