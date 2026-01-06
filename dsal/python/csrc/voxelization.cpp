/**
 * api align with
 * https://github.com/open-mmlab/mmdetection3d/blob/main/projects/BEVFusion/bevfusion/ops/voxel/src/voxelization.h
 */

#include "xpytorch.hpp"

int64_t hard_voxelize(
        const at::Tensor& points,
        at::Tensor& voxels,
        at::Tensor& coors,
        at::Tensor& num_points_per_voxel,
        const std::vector<double> voxel_size,
        const std::vector<double> coors_range,
        const int64_t max_points,
        const int64_t max_voxels,
        const int64_t NDim = 3,
        const bool deterministic = true) {
    AT_ASSERTM(deterministic == true, "Only support deterministic mode now!");
    AT_ASSERTM(NDim == 3, "Only support 3D now!");

    const int num_points = points.size(0);
    const int num_features = points.size(1);

    const float voxel_x = voxel_size[0];
    const float voxel_y = voxel_size[1];
    const float voxel_z = voxel_size[2];
    const float coors_x_min = coors_range[0];
    const float coors_y_min = coors_range[1];
    const float coors_z_min = coors_range[2];
    const float coors_x_max = coors_range[3];
    const float coors_y_max = coors_range[4];
    const float coors_z_max = coors_range[5];

    const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
    const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
    const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

    auto voxel_num = at::zeros({1,}, points.options().dtype(at::kInt));
    auto ctx = xmlir_rt::getXpuKernelContext();

    auto kernel = xav::cpu::hard_voxelize<float, int>;
    if (points.device().is_cuda()) {
        kernel = xav::xpu::hard_voxelize<float, int>;

        kernel(ctx,
           points.data_ptr<float>(),
           coors_x_min,
           coors_y_min,
           coors_z_min,
           voxel_x,
           voxel_y,
           voxel_z,
           grid_x,
           grid_y,
           grid_z,
           num_points,
           num_features,
           max_points,
           max_voxels,
           voxels.data_ptr<float>(),
           coors.data_ptr<int>(),
           num_points_per_voxel.data_ptr<int>(),
           nullptr,
           voxel_num.data_ptr<int>());
        return voxel_num.item<int>();
    } else {
        auto grid_idx_to_voxel_idx = -at::ones({grid_x, grid_y, grid_z}, at::kInt).to(points.device());
        kernel(ctx,
           points.data_ptr<float>(),
           coors_x_min,
           coors_y_min,
           coors_z_min,
           voxel_x,
           voxel_y,
           voxel_z,
           grid_x,
           grid_y,
           grid_z,
           num_points,
           num_features,
           max_points,
           max_voxels,
           voxels.data_ptr<float>(),
           coors.data_ptr<int>(),
           num_points_per_voxel.data_ptr<int>(),
           grid_idx_to_voxel_idx.data_ptr<int>(),
           voxel_num.data_ptr<int>());
        return voxel_num.item<int>();
    }
}

void hard_voxelize_forward(
        const at::Tensor& points,
        const at::Tensor& voxel_size,
        const  at::Tensor& coors_range,
        at::Tensor& voxels,
        at::Tensor& coors,
        at::Tensor& num_points_per_voxel,
        at::Tensor& voxel_num,
        const int64_t max_points,
        const int64_t max_voxels,
        const int64_t NDim = 3,
        const bool deterministic = true) {
    AT_ASSERTM(deterministic == true, "Only support deterministic mode now!");
    AT_ASSERTM(NDim == 3, "Only support 3D now!");

    const int num_points = points.size(0);
    const int num_features = points.size(1);

    const float voxel_x = voxel_size[0].item<float>();
    const float voxel_y = voxel_size[1].item<float>();
    const float voxel_z = voxel_size[2].item<float>();
    const float coors_x_min = coors_range[0].item<float>();
    const float coors_y_min = coors_range[1].item<float>();
    const float coors_z_min = coors_range[2].item<float>();
    const float coors_x_max = coors_range[3].item<float>();
    const float coors_y_max = coors_range[4].item<float>();
    const float coors_z_max = coors_range[5].item<float>();

    const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
    const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
    const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

    auto voxel_num_tmp = at::zeros({1,}, points.options().dtype(at::kInt));
    auto ctx = xmlir_rt::getXpuKernelContext();
    int ret = 0;

    auto kernel = xav::cpu::hard_voxelize_forward<float, int>;
    if (points.device().is_cuda()) {
        kernel = xav::xpu::hard_voxelize_forward<float, int>;
        ret = kernel(ctx,
           points.data_ptr<float>(),
           coors_x_min,
           coors_y_min,
           coors_z_min,
           voxel_x,
           voxel_y,
           voxel_z,
           grid_x,
           grid_y,
           grid_z,
           num_points,
           num_features,
           max_points,
           max_voxels,
           voxels.data_ptr<float>(),
           coors.data_ptr<int>(),
           num_points_per_voxel.data_ptr<int>(),
           nullptr,
           voxel_num_tmp.data_ptr<int>());
    } else {
        auto grid_idx_to_voxel_idx = -at::ones({grid_x, grid_y, grid_z}, at::kInt).to(points.device());
        ret = kernel(ctx,
           points.data_ptr<float>(),
           coors_x_min,
           coors_y_min,
           coors_z_min,
           voxel_x,
           voxel_y,
           voxel_z,
           grid_x,
           grid_y,
           grid_z,
           num_points,
           num_features,
           max_points,
           max_voxels,
           voxels.data_ptr<float>(),
           coors.data_ptr<int>(),
           num_points_per_voxel.data_ptr<int>(),
           grid_idx_to_voxel_idx.data_ptr<int>(),
           voxel_num_tmp.data_ptr<int>());
    }
    int num = voxel_num_tmp.item<int>();
    voxel_num.fill_(num);
    assert(ret == 0);
}

void dynamic_voxelize_forward(
        const at::Tensor& points,
        const at::Tensor& voxel_size,
        const  at::Tensor& coors_range,
        at::Tensor& coors,
        const int64_t NDim = 3) {
    AT_ASSERTM(NDim == 3, "Only support 3D now!");

    const int num_points = points.size(0);
    const int num_features = points.size(1);

    const float voxel_x = voxel_size[0].item<float>();
    const float voxel_y = voxel_size[1].item<float>();
    const float voxel_z = voxel_size[2].item<float>();
    const float coors_x_min = coors_range[0].item<float>();
    const float coors_y_min = coors_range[1].item<float>();
    const float coors_z_min = coors_range[2].item<float>();
    const float coors_x_max = coors_range[3].item<float>();
    const float coors_y_max = coors_range[4].item<float>();
    const float coors_z_max = coors_range[5].item<float>();

    const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
    const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
    const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

    auto kernel = xav::cpu::dynamic_voxelize<float, int>;
    if (points.device().is_cuda()) {
        kernel = xav::xpu::dynamic_voxelize<float, int>;
    }

    auto ctx = xmlir_rt::getXpuKernelContext();
    int ret = kernel(ctx,
           points.data_ptr<float>(),
           coors.data_ptr<int>(),
           coors_x_min,
           coors_y_min,
           coors_z_min,
           voxel_x,
           voxel_y,
           voxel_z,
           grid_x,
           grid_y,
           grid_z,
           num_points,
           num_features);
    assert(ret == 0);
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("hard_voxelize", &hard_voxelize);
}
TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("hard_voxelize_forward", &hard_voxelize_forward);
}
TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("dynamic_voxelize_forward", &dynamic_voxelize_forward);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("hard_voxelize", &hard_voxelize);
}
TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("hard_voxelize_forward", &hard_voxelize_forward);
}
TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("dynamic_voxelize_forward", &dynamic_voxelize_forward);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
            "hard_voxelize(Tensor points, Tensor(a!) voxels, Tensor(b!) coors, Tensor(c!) num_points_per_voxel, "
            "float[] voxel_size, float[] coors_range, int max_points, int max_voxels, int NDim=3, bool "
            "deterministic=True) -> int"));
}
TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
            "hard_voxelize_forward(Tensor points, Tensor voxel_size, Tensor coors_range, Tensor(a!) voxels, Tensor(b!) coors, Tensor(c!) num_points_per_voxel, "
            "Tensor(d!) voxel_num, int max_points, int max_voxels, int NDim=3, bool "
            "deterministic=True) -> ()"));
}
TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
            "dynamic_voxelize_forward(Tensor points, Tensor voxel_size, Tensor coors_range, "
            "Tensor(a!) coors, int NDim=3) -> ()"));
}


