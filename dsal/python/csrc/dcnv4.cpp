/**
 * api align with https://github.com/OpenGVLab/DCNv4
 */

#include "xpytorch.hpp"

/**
 * value:  (B, Hin, Win, G * D)
 * offset: (B, Hout, Wout, G * k*k*3) -- { k*k*(x,y) | k*k*(w) }
 * output: (B, Hout, Wout, G * D)
 */
using scalar_t = float;

at::Tensor dcnv4_forward(
        const at::Tensor& value,
        const at::Tensor& p_offset,
        const int64_t kernel_h,
        const int64_t kernel_w,
        const int64_t stride_h,
        const int64_t stride_w,
        const int64_t pad_h,
        const int64_t pad_w,
        const int64_t dilation_h,
        const int64_t dilation_w,
        const int64_t group,
        const int64_t group_channels,
        const double offset_scale,
        const int64_t im2col_step,
        const int64_t remove_center,
        const int64_t d_stride,
        const int64_t block_thread,
        const bool softmax) {
    AT_ASSERTM(softmax == false, "softmax is not supported");
    AT_ASSERTM(kernel_h == kernel_w, "dcnv4 only support square kernel");
    AT_ASSERTM(offset_scale == 1.0, "offset_scale is not supported");
    AT_ASSERTM(remove_center == 0, "remove center is not supported");
    AT_ASSERTM(value.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(p_offset.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(value.scalar_type() == at::ScalarType::Float, "only support float32");
    AT_ASSERTM(p_offset.scalar_type() == at::ScalarType::Float, "only support float32");

    const int batch = value.size(0);
    const int height_in = value.size(1);
    const int width_in = value.size(2);
    const int channels = value.size(3);
    const int padded_offset_dim = p_offset.size(3);

    const int height_out = (height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    AT_ASSERTM(padded_offset_dim % group == 0, "offset dim is not divisible by group");
    AT_ASSERTM(
            channels == (group * group_channels),
            "Input channels and group times group channels wont match: (%d vs %d).",
            channels,
            group * group_channels);

    auto ctx = xmlir_rt::getXpuKernelContext();

    auto output = at::zeros({batch, height_out, width_out, group * group_channels}, value.options());

    if (value.device().is_cuda()) {
        xav::xpu::dcnv4_im2col(
                ctx,
                value.data_ptr<scalar_t>(),
                p_offset.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w,
                group,
                group_channels,
                batch,
                height_in,
                width_in,
                height_out,
                width_out,
                padded_offset_dim);
    } else {
        xav::cpu::dcnv4_im2col<float>(
                ctx,
                value.data_ptr<scalar_t>(),
                p_offset.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w,
                group,
                group_channels,
                batch,
                height_in,
                width_in,
                height_out,
                width_out,
                1,    // offset_scale
                0,    // remove_center
                block_thread,
                softmax,
                padded_offset_dim);
    }

    return output;
}

/**
 * input:
 * value:       (B, Hin, Win, G, D)
 * offset:      (B, Hout, Wout, G, k*k*3) -- { k*k*(x,y) | k*k*(w) }
 * grad_output: (B, Hout, Wout, G, D)
 *
 * output:
 * grad_input:  (B, Hout, Wout, G, D)
 * grad_offset: (B, Hout, Wout, G, k*k*3)
 */
std::vector<at::Tensor> dcnv4_backward(
        const at::Tensor& value,
        const at::Tensor& p_offset,
        const int64_t kernel_h,
        const int64_t kernel_w,
        const int64_t stride_h,
        const int64_t stride_w,
        const int64_t pad_h,
        const int64_t pad_w,
        const int64_t dilation_h,
        const int64_t dilation_w,
        const int64_t group,
        const int64_t group_channels,
        const double offset_scale,
        const int64_t im2col_step,
        const at::Tensor& grad_output,
        const int64_t remove_center,
        const int64_t d_stride,
        const int64_t block_thread,
        const bool softmax) {
    AT_ASSERTM(softmax == false, "softmax is not supported");
    AT_ASSERTM(kernel_h == kernel_w, "dcnv4 only support square kernel");
    AT_ASSERTM(offset_scale == 1.0, "offset_scale is not supported");
    AT_ASSERTM(remove_center == 0, "remove center is not supported");
    AT_ASSERTM(value.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(p_offset.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(value.scalar_type() == at::ScalarType::Float, "only support float32");
    AT_ASSERTM(p_offset.scalar_type() == at::ScalarType::Float, "only support float32");
    AT_ASSERTM(grad_output.scalar_type() == at::ScalarType::Float, "only support float32");

    auto ctx = xmlir_rt::getXpuKernelContext();

    const int batch = value.size(0);
    const int height_in = value.size(1);
    const int width_in = value.size(2);
    const int channels = value.size(3);
    const int padded_offset_dim = p_offset.size(3);
    // assert(padded_offset_dim % 8 == 0);

    const int height_out = (height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    AT_ASSERTM(padded_offset_dim % group == 0, "offset dim is not divisible by group");
    AT_ASSERTM(
            channels == (group * group_channels),
            "Input channels and group times group channels wont match: (%d vs %d).",
            channels,
            group * group_channels);

    auto grad_input = at::zeros_like(value, value.dtype());
    auto grad_offset = at::zeros_like(p_offset, p_offset.dtype());

    if (value.device().is_cuda()) {
        xav::xpu::dcnv4_col2im(
                ctx,
                value.data_ptr<scalar_t>(),
                p_offset.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(),
                grad_offset.data_ptr<scalar_t>(),
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w,
                group,
                group_channels,
                batch,
                height_in,
                width_in,
                height_out,
                width_out,
                padded_offset_dim);
    } else {
        xav::cpu::dcnv4_col2im<float>(
                ctx,
                value.data_ptr<scalar_t>(),
                p_offset.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w,
                group,
                group_channels,
                batch,
                height_in,
                width_in,
                height_out,
                width_out,
                1,    // offset_scale
                0,    // remove_center
                grad_input.data_ptr<scalar_t>(),
                grad_offset.data_ptr<scalar_t>(),
                block_thread,
                softmax,
                padded_offset_dim);
    }

    return {grad_input, grad_offset};
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("dcnv4_forward", &dcnv4_forward);
    m.impl("dcnv4_backward", &dcnv4_backward);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("dcnv4_forward", &dcnv4_forward);
    m.impl("dcnv4_backward", &dcnv4_backward);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
            "dcnv4_forward(Tensor value, Tensor p_offset, int kernel_h, int kernel_w, int stride_h, int "
            "stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int group, int group_channels, float "
            "offset_scale, int im2col_step, int remove_center, int d_stride, int block_thread, bool softmax) -> "
            "Tensor"));

    m.def(TORCH_SELECTIVE_SCHEMA(
            "dcnv4_backward(Tensor value, Tensor p_offset, int kernel_h, int kernel_w, int stride_h, int stride_w, "
            "int pad_h, int pad_w, int dilation_h, int dilation_w, int group, int group_channels, float "
            "offset_scale, int im2col_step, Tensor grad_output, int remove_center, int d_stride, int block_thread, "
            "bool softmax) -> Tensor[]"));
}
