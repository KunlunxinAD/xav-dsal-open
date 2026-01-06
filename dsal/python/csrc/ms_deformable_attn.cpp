#include "xpytorch.hpp"

at::Tensor ms_deform_attn_forward(
        const at::Tensor& value,
        const at::Tensor& spatial_shapes,
        const at::Tensor& level_start_index,
        const at::Tensor& sampling_loc,
        const at::Tensor& attn_weight,
        const int64_t im2col_step) {
    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    auto output = at::zeros({batch, num_query, num_heads * channels}, value.options());

    float* value_ptr = value.data_ptr<float>();
    int64_t* spatial_shapes_ptr = spatial_shapes.data_ptr<int64_t>();
    int64_t* level_start_index_ptr = level_start_index.data_ptr<int64_t>();
    float* sampling_loc_ptr = sampling_loc.data_ptr<float>();
    float* attn_weight_ptr = attn_weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    auto ctx = xmlir_rt::getXpuKernelContext();

    int ret = 0;

    if (value.device().is_cuda()) {
        // xpu implementation
        ret = xav::xpu::ms_deformable_im2col(
                ctx,
                value_ptr,
                sampling_loc_ptr,
                attn_weight_ptr,
                /*data_col*/ output_ptr,
                spatial_shapes_ptr,
                spatial_shapes.numel(),
                level_start_index_ptr,
                level_start_index.numel(),
                batch,
                spatial_size,
                num_heads,
                channels,
                num_levels,
                num_query,
                num_point);
    } else {
        // cpu implementation
        ret = xav::cpu::ms_deformable_im2col(
                ctx,
                value_ptr,
                sampling_loc_ptr,
                attn_weight_ptr,
                /*data_col*/ output_ptr,
                spatial_shapes_ptr,
                spatial_shapes.numel(),
                level_start_index_ptr,
                level_start_index.numel(),
                batch,
                spatial_size,
                num_heads,
                channels,
                num_levels,
                num_query,
                num_point);
    }
    assert(ret == 0 && "xav_dsal::ms_deformable_im2col failed");

    return output;
}

void ms_deform_attn_backward(
        const at::Tensor& value,
        const at::Tensor& spatial_shapes,
        const at::Tensor& level_start_index,
        const at::Tensor& sampling_loc,
        const at::Tensor& attn_weight,
        const at::Tensor& grad_output,
        at::Tensor& grad_value,
        at::Tensor& grad_sampling_loc,
        at::Tensor& grad_attn_weight,
        const int64_t im2col_step) {
    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = grad_output.size(1);

    const int num_point = sampling_loc.size(4);

    float* grad_output_ptr = grad_output.data_ptr<float>();
    float* value_ptr = value.data_ptr<float>();
    int64_t* spatial_shapes_ptr = spatial_shapes.data_ptr<int64_t>();
    int64_t* level_start_index_ptr = level_start_index.data_ptr<int64_t>();
    float* sampling_loc_ptr = sampling_loc.data_ptr<float>();
    float* attn_weight_ptr = attn_weight.data_ptr<float>();
    float* grad_value_ptr = grad_value.data_ptr<float>();
    float* grad_sampling_loc_ptr = grad_sampling_loc.data_ptr<float>();
    float* grad_attn_weight_ptr = grad_attn_weight.data_ptr<float>();

    auto ctx = xmlir_rt::getXpuKernelContext();

    int ret = 0;

    if (value.device().is_cuda()) {
        // xpu implementation
        ret = xav::xpu::ms_deformable_col2im(
                ctx,
                grad_output_ptr,
                value_ptr,
                sampling_loc_ptr,
                attn_weight_ptr,
                grad_value_ptr,
                grad_sampling_loc_ptr,
                grad_attn_weight_ptr,
                spatial_shapes_ptr,
                spatial_shapes.numel(),
                level_start_index_ptr,
                level_start_index.numel(),
                batch,
                spatial_size,
                num_heads,
                channels,
                num_levels,
                num_query,
                num_point);
    } else {
        // cpu implementation
        ret = xav::cpu::ms_deformable_col2im(
                ctx,
                grad_output_ptr,
                value_ptr,
                sampling_loc_ptr,
                attn_weight_ptr,
                grad_value_ptr,
                grad_sampling_loc_ptr,
                grad_attn_weight_ptr,
                spatial_shapes_ptr,
                spatial_shapes.numel(),
                level_start_index_ptr,
                level_start_index.numel(),
                batch,
                spatial_size,
                num_heads,
                channels,
                num_levels,
                num_query,
                num_point);
    }

    assert(r == 0 && "xav_dsal::ms_deformable_col2im failed");
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("ms_deform_attn_forward", &ms_deform_attn_forward);
    m.impl("ms_deform_attn_backward", &ms_deform_attn_backward);
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("ms_deform_attn_forward", &ms_deform_attn_forward);
    m.impl("ms_deform_attn_backward", &ms_deform_attn_backward);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("ms_deform_attn_forward(Tensor value, Tensor spatial_shapes, Tensor "
                                 "level_start_index, Tensor sampling_loc, "
                                 "Tensor attn_weight, int im2col_step) -> Tensor"));

    m.def(TORCH_SELECTIVE_SCHEMA(
            "ms_deform_attn_backward(Tensor value, Tensor spatial_shapes, Tensor level_start_index, Tensor "
            "sampling_loc, Tensor attn_weight, Tensor grad_output, Tensor(a!) grad_value, Tensor(b!) "
            "grad_sampling_loc, Tensor(c!) grad_attn_weight, int im2col_step) -> ()"));
}
