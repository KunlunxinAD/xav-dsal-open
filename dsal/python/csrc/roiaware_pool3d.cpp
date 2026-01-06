#include "xpytorch.hpp"

void roiaware_pool3d_forward(const at::Tensor& rois, const at::Tensor& pts, const at::Tensor& pts_feature, 
    at::Tensor& argmax, at::Tensor& pts_idx_of_voxels, at::Tensor& pooled_features, int64_t pool_method) {
    auto ctx = xmlir_rt::getXpuKernelContext();
    AT_ASSERTM((pool_method == 0 || pool_method == 1), "pool_method must be 0 or 1");
    TORCH_CHECK(rois.is_contiguous(), " must be contiguous");
    TORCH_CHECK(pts.is_contiguous(), " must be contiguous");
    TORCH_CHECK(pts_feature.is_contiguous(), " must be contiguous");
    TORCH_CHECK(argmax.is_contiguous(), " must be contiguous");
    TORCH_CHECK(pts_idx_of_voxels.is_contiguous(), " must be contiguous");
    TORCH_CHECK(pooled_features.is_contiguous(), " must be contiguous");

    AT_ASSERTM(rois.dim() == 2, "rois must be 2 dim");
    AT_ASSERTM(pts.dim() == 2, "pts must be 2 dim");
    AT_ASSERTM(pts_feature.dim() == 2, "pts_feature must be 2 dim");
    AT_ASSERTM(argmax.dim() == 5, "argmax must be 5 dim");
    AT_ASSERTM(pts_idx_of_voxels.dim() == 5, "pts_idx_of_voxels must be 5 dim");
    AT_ASSERTM(pooled_features.dim() == 5, "pooled_features must be 5 dim");


    AT_ASSERTM(rois.size(1) == 7, "rois.size(1) must = 7, which is [x, y, z, dx, dy, dz, heading]");
    AT_ASSERTM(pts.size(1) == 3, "pts.size(1) must = 3, which is [x, y, z]");
    AT_ASSERTM(argmax.size(4) == pooled_features.size(4), "argmax.size(4) must == pooled_features.size(4), which is channel");
    AT_ASSERTM(pts.size(0) == pts_feature.size(0), "pts.size(0) must == pts_feature.size(0), which is npoints");
    
    bool boxes_num_flag = (rois.size(0) == argmax.size(0)) &&
                            (argmax.size(0) == pts_idx_of_voxels.size(0)) &&
                            (pts_idx_of_voxels.size(0) == pooled_features.size(0));
    bool channels_flag = (pts_feature.size(1) == argmax.size(4)) && (argmax.size(4) == pooled_features.size(4));
    bool out_x_flag = (argmax.size(1) == pts_idx_of_voxels.size(1)) && 
                        (pts_idx_of_voxels.size(1) == pooled_features.size(1));
    bool out_y_flag = (argmax.size(2) == pts_idx_of_voxels.size(2)) && 
                        (pts_idx_of_voxels.size(2) == pooled_features.size(2));
    bool out_z_flag = (argmax.size(3) == pts_idx_of_voxels.size(3)) && 
                        (pts_idx_of_voxels.size(3) == pooled_features.size(3));
    AT_ASSERTM(boxes_num_flag, "rois.size(0) must == argmax.size(0) == pts_idx_of_voxels.size(0) == \
                pooled_features.size(0), which is boxes num");
    AT_ASSERTM(channels_flag, "pts_feature.size(1) == argmax.size(4) == pooled_features.size(4), which is channels");
    AT_ASSERTM(out_x_flag, "argmax.size(1) == pts_idx_of_voxels.size(1) == pooled_features.size(1), which is out_x");
    AT_ASSERTM(out_y_flag, "argmax.size(2) == pts_idx_of_voxels.size(2) == pooled_features.size(2), which is out_y");
    AT_ASSERTM(out_z_flag, "argmax.size(3) == pts_idx_of_voxels.size(3) == pooled_features.size(3), which is out_z");

    int boxes_num = rois.size(0);
    int pts_num = pts.size(0);
    int channels = pts_feature.size(1);
    int max_pts_each_voxels = pts_idx_of_voxels.size(4);
    int out_x = pts_idx_of_voxels.size(1);
    int out_y = pts_idx_of_voxels.size(2);
    int out_z = pts_idx_of_voxels.size(3);
    AT_ASSERTM(pts_num <= 1000, "Pts_num must <= 1000");
    AT_ASSERTM(max_pts_each_voxels <= pts_num, "max_pts_each_voxels must <= pts_num");
    
    int ret = 0;
    if (rois.device().is_cuda()) {
        ret = xav::xpu::roiaware_pool3d<float>(
            ctx,
            rois.data_ptr<float>(),
            pts.data_ptr<float>(),
            pts_feature.data_ptr<float>(),
            pooled_features.data_ptr<float>(),
            pool_method,
            argmax.data_ptr<int>(),
            pts_idx_of_voxels.data_ptr<int>(),
            boxes_num,
            pts_num,
            channels,
            max_pts_each_voxels,
            out_x,
            out_y,
            out_z
        );
    } else {
        ret = xav::cpu::roiaware_pool3d<float>(
            ctx,
            rois.data_ptr<float>(),
            pts.data_ptr<float>(),
            pts_feature.data_ptr<float>(),
            pooled_features.data_ptr<float>(),
            pool_method,
            argmax.data_ptr<int>(),
            pts_idx_of_voxels.data_ptr<int>(),
            boxes_num,
            pts_num,
            channels,
            max_pts_each_voxels,
            out_x,
            out_y,
            out_z
        );
    }
    assert(ret == 0);
}

void roiaware_pool3d_grad(const at::Tensor& pts_idx_of_voxels, const at::Tensor& argmax, const at::Tensor& grad_out, 
                        at::Tensor& grad_in, int64_t pool_method) {
    auto ctx = xmlir_rt::getXpuKernelContext();
    AT_ASSERTM((pool_method == 0 || pool_method == 1), "pool_method must be 0 or 1");
    TORCH_CHECK(pts_idx_of_voxels.is_contiguous(), " must be contiguous");
    TORCH_CHECK(argmax.is_contiguous(), " must be contiguous");
    TORCH_CHECK(grad_out.is_contiguous(), " must be contiguous");
    TORCH_CHECK(grad_in.is_contiguous(), " must be contiguous");

    AT_ASSERTM(pts_idx_of_voxels.dim() == 5, "pts_idx_of_voxels must be 5 dim");
    AT_ASSERTM(argmax.dim() == 5, "argmax must be 5 dim");
    AT_ASSERTM(grad_out.dim() == 5, "grad_out must be 5 dim");
    AT_ASSERTM(grad_in.dim() == 2, "grad_out must be 2 dim");

    int boxes_num = pts_idx_of_voxels.size(0);
    int out_x = pts_idx_of_voxels.size(1);
    int out_y = pts_idx_of_voxels.size(2);
    int out_z = pts_idx_of_voxels.size(3);
    int max_pts_each_voxel = pts_idx_of_voxels.size(4);  // index 0 is the counter
    int channels = grad_out.size(4);
    int npoints = grad_in.size(0);

    int ret = 0;
    if (pts_idx_of_voxels.device().is_cuda()) {
        ret = xav::xpu::roiaware_pool3d_grad<float>(
            ctx,
            boxes_num,
            out_x,
            out_y,
            out_z,
            channels,
            max_pts_each_voxel,
            npoints,
            pts_idx_of_voxels.data_ptr<int>(),
            argmax.data_ptr<int>(),
            grad_out.data_ptr<float>(),
            grad_in.data_ptr<float>(),
            pool_method);
    } else {
        ret = xav::cpu::roiaware_pool3d_grad<float>(
            ctx,
            boxes_num,
            out_x,
            out_y,
            out_z,
            channels,
            max_pts_each_voxel,
            npoints,
            pts_idx_of_voxels.data_ptr<int>(),
            argmax.data_ptr<int>(),
            grad_out.data_ptr<float>(),
            grad_in.data_ptr<float>(),
            pool_method);
    }
    assert(ret == 0);
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("roiaware_pool3d_forward", &roiaware_pool3d_forward);
    m.impl("roiaware_pool3d_grad", &roiaware_pool3d_grad);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("roiaware_pool3d_forward", &roiaware_pool3d_forward);
    m.impl("roiaware_pool3d_grad", &roiaware_pool3d_grad);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("roiaware_pool3d_forward(Tensor rois, Tensor pts, Tensor pts_feature, "
        "Tensor(a!) argmax, Tensor(b!) pts_idx_of_voxels, Tensor(c!) pooled_features, int pool_method) -> ()"));
    m.def(TORCH_SELECTIVE_SCHEMA("roiaware_pool3d_grad(Tensor pts_idx_of_voxels, Tensor argmax, Tensor grad_out, "
        "Tensor(a!) grad_in, int pool_method) -> ()"));
}