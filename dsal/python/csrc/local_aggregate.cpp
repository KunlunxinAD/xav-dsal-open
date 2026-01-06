#include "xpytorch.hpp"

#define NUM_CHANNELS 18

std::tuple<int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> local_aggregate_forward(
        const at::Tensor& pts,
        const at::Tensor& points_int,
        const at::Tensor& means3D,
        const at::Tensor& means3D_int,
        const at::Tensor& opacity,
        const at::Tensor& semantics,
        const at::Tensor& radii,
        const at::Tensor& cov3D,
        int64_t _H,
        int64_t _W,
        int64_t _D) {
    int ret = 0;
    auto xpu_ctx = xmlir_rt::getXpuKernelContext();

    const int P = means3D.size(0);
    const int N = pts.size(0);
    const int H = static_cast<int>(_H);
    const int W = static_cast<int>(_W);
    const int D = static_cast<int>(_D);

    int num_rendered;
    at::Tensor out_logits = at::full({N, NUM_CHANNELS}, 0.0, pts.options().dtype(at::kFloat));
    at::Tensor point_offsets = at::empty({P}, pts.options().dtype(at::kInt));
    at::Tensor tiles_touched = at::empty({P}, pts.options().dtype(at::kInt));

    int* means3D_int_ptr = means3D_int.data_ptr<int>();
    int* radii_ptr = radii.data_ptr<int>();
    int* tiles_touched_ptr = tiles_touched.data_ptr<int>();
    int* point_offsets_ptr = point_offsets.data_ptr<int>();

    auto kernel_preprocess = xav::cpu::local_aggregate_preprocess;
    if (pts.device().is_cuda()) {
        kernel_preprocess = xav::xpu::local_aggregate_preprocess;
    }
    kernel_preprocess(
            xpu_ctx, means3D_int_ptr, radii_ptr, tiles_touched_ptr, point_offsets_ptr, P, H, W, D, num_rendered);
#if 0
    at::Tensor point_list_map_unsorted = at::empty({num_rendered, 2}, pts.options().dtype(at::kInt));
    at::Tensor point_list_map = at::empty({num_rendered, 2}, pts.options().dtype(at::kInt));

    int* point_list_map_unsorted_ptr = point_list_map_unsorted.data_ptr<int>();
    int* point_list_map_ptr = point_list_map.data_ptr<int>();
    if (pts.device().is_cuda()) {
        xav::xpu::local_aggregate_dup_keys(
                xpu_ctx,
                means3D_int_ptr,
                radii_ptr,
                point_offsets_ptr,
                point_list_map_unsorted_ptr,
                P,
                H,
                W,
                D);
        at::Tensor sorted_indices = at::empty({num_rendered}, pts.options().dtype(at::kInt));
        static auto op = at::Dispatcher::singleton()
                                 .findSchemaOrThrow("custom_ops::sort_2d_stable", "")
                                 .typed<std::tuple<at::Tensor&, at::Tensor&>(
                                         at::Tensor const&, bool, long, bool, at::Tensor&, at::Tensor&)>();
        op.call(point_list_map_unsorted, true, 0, false, point_list_map, sorted_indices);
    } else {
        xav::cpu::local_aggregate_dup_keys(
                xpu_ctx,
                means3D_int_ptr,
                radii_ptr,
                point_offsets_ptr,
                point_list_map_unsorted_ptr,
                P,
                H,
                W,
                D);
        at::Tensor keys = point_list_map_unsorted.select(1, 0);
        auto [sorted_keys, indices] = keys.sort(0, false);
        point_list_map = point_list_map_unsorted.index_select(0, indices);
    }
#else
    at::Tensor point_list_keys_unsorted = at::empty({num_rendered}, pts.options().dtype(at::kInt));
    at::Tensor point_list_unsorted = at::empty({num_rendered}, pts.options().dtype(at::kInt));
    at::Tensor point_list_keys = at::empty({num_rendered}, pts.options().dtype(at::kInt));
    at::Tensor point_list = at::empty({num_rendered}, pts.options().dtype(at::kInt));
    at::Tensor ranges = at::zeros({H * W * D, 2}, pts.options().dtype(at::kInt));

    int* point_list_keys_unsorted_ptr = point_list_keys_unsorted.data_ptr<int>();
    int* point_list_unsorted_ptr = point_list_unsorted.data_ptr<int>();

    auto kernel_dup_keys = xav::cpu::local_aggregate_dup_keys;
    if (pts.device().is_cuda()) {
        kernel_dup_keys = xav::xpu::local_aggregate_dup_keys;
    }
    kernel_dup_keys(
            xpu_ctx,
            means3D_int_ptr,
            radii_ptr,
            point_offsets_ptr,
            point_list_keys_unsorted_ptr,
            point_list_unsorted_ptr,
            P,
            H,
            W,
            D);
    at::Tensor sorted_index = at::empty({num_rendered}, pts.options().dtype(at::kInt));
    std::tie(point_list_keys, sorted_index) = point_list_keys_unsorted.sort(0, false);
    point_list = point_list_unsorted.index_select(0, sorted_index);
#endif
    float* pts_ptr = pts.data_ptr<float>();
    int* points_int_ptr = points_int.data_ptr<int>();
    int* point_list_keys_ptr = point_list_keys.data_ptr<int>();
    int* point_list_ptr = point_list.data_ptr<int>();
    float* means3D_ptr = means3D.data_ptr<float>();
    float* opacity_ptr = opacity.data_ptr<float>();
    float* semantics_ptr = semantics.data_ptr<float>();
    float* cov3D_ptr = cov3D.data_ptr<float>();
    int* ranges_ptr = ranges.data_ptr<int>();
    float* out_ptr = out_logits.data_ptr<float>();

    auto kernel_process = xav::cpu::local_aggregate_process;
    if (pts.device().is_cuda()) {
        kernel_process = xav::xpu::local_aggregate_process;
    }
    kernel_process(
            xpu_ctx,
            pts_ptr,
            points_int_ptr,
            point_list_keys_ptr,
            point_list_ptr,
            means3D_ptr,
            opacity_ptr,
            semantics_ptr,
            cov3D_ptr,
            ranges_ptr,
            out_ptr,
            N,
            H,
            W,
            D,
            num_rendered);
    ranges = ranges.flatten();

    return {num_rendered, point_offsets, point_list_keys_unsorted, point_list_unsorted, ranges, out_logits};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> local_aggregate_backward(
        const at::Tensor& point_offsets,
        const at::Tensor& point_list_keys_unsorted,
        const at::Tensor& means3D,
        const at::Tensor& pts,
        const at::Tensor& points_int,
        const at::Tensor& cov3D,
        const at::Tensor& opacities,
        const at::Tensor& semantics,
        const at::Tensor& out_grad,
        int64_t _H,
        int64_t _W,
        int64_t _D,
        int64_t _R) {
    int ret = 0;
    auto xpu_ctx = xmlir_rt::getXpuKernelContext();

    const int H = static_cast<int>(_H);
    const int W = static_cast<int>(_W);
    const int D = static_cast<int>(_D);
    const int R = static_cast<int>(_R);
    const int P = means3D.size(0);
    const int N = pts.size(0);

    at::Tensor means3D_grad = at::zeros({P, 3}, means3D.options());
    at::Tensor opacity_grad = at::zeros({P}, means3D.options());
    at::Tensor semantics_grad = at::zeros({P, NUM_CHANNELS}, means3D.options());
    at::Tensor cov3D_grad = at::zeros({P, 6}, means3D.options());

    at::Tensor voxel2pts = at::full({H * W * D}, -1, means3D.options().dtype(at::kInt));

    auto kernel = xav::cpu::local_aggregate_backward;
    if (pts.device().is_cuda()) {
        kernel = xav::xpu::local_aggregate_backward;
    }
    kernel(xpu_ctx,
           point_offsets.data<int>(),
           point_list_keys_unsorted.data<int>(),
           points_int.data<int>(),
           pts.data<float>(),
           means3D.data<float>(),
           cov3D.data<float>(),
           opacities.data<float>(),
           semantics.data<float>(),
           out_grad.data<float>(),
           voxel2pts.data<int>(),
           means3D_grad.data<float>(),
           opacity_grad.data<float>(),
           semantics_grad.data<float>(),
           cov3D_grad.data<float>(),
           P,
           R,
           N,
           H,
           W,
           D);
    return {means3D_grad, opacity_grad, semantics_grad, cov3D_grad, voxel2pts};
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("local_aggregate_forward", &local_aggregate_forward);
    m.impl("local_aggregate_backward", &local_aggregate_backward);
}
TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("local_aggregate_forward", &local_aggregate_forward);
    m.impl("local_aggregate_backward", &local_aggregate_backward);
}
TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("local_aggregate_forward(Tensor pts, Tensor points_int, Tensor means3D, Tensor "
                                 "means3D_int, Tensor opacity, Tensor semantics, Tensor radii, Tensor cov3D, int H, "
                                 "int W, int D) -> (int, Tensor, Tensor, Tensor, Tensor, Tensor)"));
    m.def(TORCH_SELECTIVE_SCHEMA(
            "local_aggregate_backward(Tensor point_offsets, Tensor point_list_keys_unsorted, Tensor means3D, Tensor "
            "pts, Tensor points_int, Tensor cov3D, Tensor opacities, Tensor "
            "semantics, Tensor out_grad, int H, int W, int "
            "D, int R) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"));
}
