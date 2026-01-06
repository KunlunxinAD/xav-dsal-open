#include "xpytorch.hpp"

void sigmoid_focal_loss_forward(
        const at::Tensor& input,
        const at::Tensor& target,
        const at::Tensor& weight,
        at::Tensor& output,
        double gamma,
        double alpha) {
    TORCH_CHECK(
            input.scalar_type() == at::ScalarType::Float, "sigmoid_focal_loss only supports input with float32 dtype");
    TORCH_CHECK(
            output.scalar_type() == at::ScalarType::Float,
            "sigmoid_focal_loss only supports output with float32 dtype");
    TORCH_CHECK(
            target.scalar_type() == at::ScalarType::Long, "sigmoid_focal_loss only supports target with long dtype");

    float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    int64_t* target_ptr = target.data_ptr<int64_t>();
    float* weight_ptr = nullptr;
    weight_ptr = weight.data_ptr<float>();

    auto ctx = xmlir_rt::getXpuKernelContext();

    xav::xpu::sigmoid_focal_loss(
            ctx,
            input_ptr,
            target_ptr,
            weight_ptr,
            output_ptr,
            gamma,
            alpha,
            input.numel(),
            target.numel(),
            input.size(1));
}

void sigmoid_focal_loss_backward(
        const at::Tensor& input,
        const at::Tensor& target,
        const at::Tensor& weight,
        at::Tensor& grad_input,
        double gamma,
        double alpha) {
    TORCH_CHECK(
            input.scalar_type() == at::ScalarType::Float,
            "sigmoid_focal_loss_backward only supports input with float32 dtype");

    TORCH_CHECK(
            grad_input.scalar_type() == at::ScalarType::Float,
            "sigmoid_focal_loss_backward only supports grad_input with "
            "float32 dtype");

    TORCH_CHECK(
            target.scalar_type() == at::ScalarType::Long,
            "sigmoid_focal_loss_backward only supports target with long dtype");

    float* input_ptr = input.data_ptr<float>();
    float* grad_input_ptr = grad_input.data_ptr<float>();
    int64_t* target_ptr = target.data_ptr<int64_t>();
    float* weight_ptr = nullptr;
    weight_ptr = weight.data_ptr<float>();

    auto ctx = xmlir_rt::getXpuKernelContext();

    xav::xpu::sigmoid_focal_loss_grad(
            ctx,
            input_ptr,
            target_ptr,
            weight_ptr,
            grad_input_ptr,
            gamma,
            alpha,
            input.numel(),
            target.numel(),
            input.size(1));
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("sigmoid_focal_loss_forward", &sigmoid_focal_loss_forward);
    m.impl("sigmoid_focal_loss_backward", &sigmoid_focal_loss_backward);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
            "sigmoid_focal_loss_forward(Tensor input, Tensor target, Tensor weight, Tensor(a!) output, float gamma, "
            "float alpha) -> ()"));

    m.def(TORCH_SELECTIVE_SCHEMA("sigmoid_focal_loss_backward(Tensor input, Tensor target, Tensor weight, Tensor(a!) "
                                 "grad_input, float gamma, float alpha) -> ()"));
}
