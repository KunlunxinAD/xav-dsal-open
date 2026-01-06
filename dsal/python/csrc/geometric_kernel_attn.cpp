#pragma once
#include "xpytorch.hpp"

#define AT_DISPATCH_CASE_FLOAT32(...) AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)

#define AT_DISPATCH_FLOAT32_ONLY(TYPE, NAME, ...) AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOAT32(__VA_ARGS__))

at::Tensor geometric_kernel_attn_cuda_forward(
        const at::Tensor& value,
        const at::Tensor& spatial_shapes,
        const at::Tensor& level_start_index,
        const at::Tensor& sampling_loc,
        const at::Tensor& attn_weight,
        const int64_t im2col_step) {
    AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    AT_ASSERTM(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    AT_ASSERTM(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    AT_ASSERTM(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");

    AT_ASSERTM(value.device().is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(spatial_shapes.device().is_cuda(), "spatial_shapes must be a CUDA tensor");
    AT_ASSERTM(level_start_index.device().is_cuda(), "level_start_index must be a CUDA tensor");
    AT_ASSERTM(sampling_loc.device().is_cuda(), "sampling_loc must be a CUDA tensor");
    AT_ASSERTM(attn_weight.device().is_cuda(), "attn_weight must be a CUDA tensor");

    auto ctx = xmlir_rt::getXpuKernelContext();

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, static_cast<int>(im2col_step));

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto output = at::zeros({batch, num_query, num_heads, channels}, value.options());

    const int batch_n = im2col_step_;
    auto output_n = output.view({batch / im2col_step_, batch_n, num_query, num_heads, channels});
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    for (int n = 0; n < batch / im2col_step_; ++n) {
        auto columns = output_n.select(0, n);

        AT_DISPATCH_FLOAT32_ONLY(value.scalar_type(), "multiscale_kernel_attn_forward_cuda", ([&] {
                                     int ret = xav::xpu::geometric_kernel_attn(
                                             ctx,
                                             value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
                                             spatial_shapes.data_ptr<int64_t>(),
                                             level_start_index.data_ptr<int64_t>(),
                                             sampling_loc.data_ptr<int64_t>() + n * im2col_step_ * per_sample_loc_size,
                                             attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                                             batch_n,
                                             spatial_size,
                                             num_heads,
                                             channels,
                                             num_levels,
                                             num_query,
                                             num_point,
                                             columns.data_ptr<scalar_t>());
                                 }));
    }

    output = output.view({batch, num_query, num_heads * channels});

    return output;
}

std::vector<at::Tensor> geometric_kernel_attn_cuda_backward(
        const at::Tensor& value,
        const at::Tensor& spatial_shapes,
        const at::Tensor& level_start_index,
        const at::Tensor& sampling_loc,
        const at::Tensor& attn_weight,
        const at::Tensor& grad_output,
        const int im2col_step) {
    AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    AT_ASSERTM(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    AT_ASSERTM(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    AT_ASSERTM(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    AT_ASSERTM(value.device().is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(spatial_shapes.device().is_cuda(), "spatial_shapes must be a CUDA tensor");
    AT_ASSERTM(level_start_index.device().is_cuda(), "level_start_index must be a CUDA tensor");
    AT_ASSERTM(sampling_loc.device().is_cuda(), "sampling_loc must be a CUDA tensor");
    AT_ASSERTM(attn_weight.device().is_cuda(), "attn_weight must be a CUDA tensor");
    AT_ASSERTM(grad_output.device().is_cuda(), "grad_output must be a CUDA tensor");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto grad_value = at::zeros_like(value);
    auto grad_attn_weight = at::zeros_like(attn_weight);

    const int batch_n = im2col_step_;
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    auto grad_output_n = grad_output.view({batch / im2col_step_, batch_n, num_query, num_heads, channels});

    auto ctx = xmlir_rt::getXpuKernelContext();

    for (int n = 0; n < batch / im2col_step_; ++n) {
        auto grad_output_g = grad_output_n.select(0, n);

        AT_DISPATCH_FLOAT32_ONLY(
                value.scalar_type(), "multiscale_kernel_attn_backward_cuda", ([&] {
                    int ret = xav::xpu::geometric_kernel_attn_grad(
                            ctx,
                            grad_output_g.data_ptr<scalar_t>(),
                            value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
                            spatial_shapes.data_ptr<int64_t>(),
                            level_start_index.data_ptr<int64_t>(),
                            sampling_loc.data_ptr<int64_t>() + n * im2col_step_ * per_sample_loc_size,
                            attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                            batch_n,
                            spatial_size,
                            num_heads,
                            channels,
                            num_levels,
                            num_query,
                            num_point,
                            grad_value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
                            grad_attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size);
                }));
    }

    return {grad_value, grad_attn_weight};
}

at::Tensor geometric_kernel_attn_forward(
        const at::Tensor& value,
        const at::Tensor& spatial_shapes,
        const at::Tensor& level_start_index,
        const at::Tensor& sampling_loc,
        const at::Tensor& attn_weight,
        const int64_t im2col_step) {
    if (value.device().is_cuda()) {
        return geometric_kernel_attn_cuda_forward(
                value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor> geometric_kernel_attn_backward(
        const at::Tensor& value,
        const at::Tensor& spatial_shapes,
        const at::Tensor& level_start_index,
        const at::Tensor& sampling_loc,
        const at::Tensor& attn_weight,
        const at::Tensor& grad_output,
        const int64_t im2col_step) {
    if (value.device().is_cuda()) {
        return geometric_kernel_attn_cuda_backward(
                value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
    }
    AT_ERROR("Not implemented on the CPU");
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("geometric_kernel_attn_forward", &geometric_kernel_attn_forward);
    m.impl("geometric_kernel_attn_backward", &geometric_kernel_attn_backward);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
            "geometric_kernel_attn_forward(Tensor value, Tensor spatial_shapes, Tensor level_start_index, Tensor "
            "sampling_loc, Tensor attn_weight, int im2col_step) -> Tensor"));

    m.def(TORCH_SELECTIVE_SCHEMA(
            "geometric_kernel_attn_backward(Tensor value, Tensor spatial_shapes, Tensor level_start_index, Tensor "
            "sampling_loc, Tensor attn_weight, Tensor grad_output, int im2col_step) -> Tensor[]"));
}
