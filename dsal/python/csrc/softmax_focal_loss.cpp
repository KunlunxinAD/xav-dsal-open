#include "xpytorch.hpp"

void softmax_focal_loss_forward(
        const at::Tensor& softmax,
        const at::Tensor& target,
        const at::optional<at::Tensor>& weight,
        at::Tensor& output,
        double gamma,
        double alpha) {
    TORCH_CHECK(
            softmax.scalar_type() == at::ScalarType::Float,
            "softmax_focal_loss only supports softmax with float32 dtype");
    TORCH_CHECK(
            output.scalar_type() == at::ScalarType::Float,
            "softmax_focal_loss only supports output with float32 dtype");
    TORCH_CHECK(
            target.scalar_type() == at::ScalarType::Long, "softmax_focal_loss only supports target with long dtype");

    auto ctx = xmlir_rt::getXpuKernelContext();

    float* softmax_ptr = softmax.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    int64_t* target_ptr = target.data_ptr<int64_t>();
    float* weight_ptr = nullptr;

    if (weight.has_value()) {
        weight_ptr = weight.value().data_ptr<float>();
    }

    if (softmax.device().is_cuda()) {
        xav::xpu::softmax_focal_loss(
                ctx, softmax.size(0), softmax_ptr, target_ptr, weight_ptr, output_ptr, gamma, alpha, softmax.size(1));
    } else {
        xav::cpu::softmax_focal_loss(
                ctx, softmax.size(0), softmax_ptr, target_ptr, weight_ptr, output_ptr, gamma, alpha, softmax.size(1));
    }
}

void softmax_focal_loss_backward(
        const at::Tensor& softmax,
        const at::Tensor& target,
        const at::optional<at::Tensor>& weight,
        at::Tensor& buff,
        at::Tensor& grad_input,
        double gamma,
        double alpha) {
    TORCH_CHECK(
            softmax.scalar_type() == at::ScalarType::Float,
            "softmax_focal_loss_backward only supports input with float32 dtype");

    TORCH_CHECK(
            grad_input.scalar_type() == at::ScalarType::Float,
            "softmax_focal_loss_backward only supports grad_input with "
            "float32 dtype");

    TORCH_CHECK(
            target.scalar_type() == at::ScalarType::Long,
            "softmax_focal_loss_backward only supports target with long dtype");

    auto ctx = xmlir_rt::getXpuKernelContext();

    float* softmax_ptr = softmax.data_ptr<float>();
    float* grad_input_ptr = grad_input.data_ptr<float>();
    int64_t* target_ptr = target.data_ptr<int64_t>();
    float* buff_ptr = buff.data_ptr<float>();
    float* weight_ptr = nullptr;

    if (weight.has_value()) {
        weight_ptr = weight.value().data_ptr<float>();
    }
    if (softmax.device().is_cuda()) {
        xav::xpu::softmax_focal_loss_grad(
                ctx,
                softmax.size(0),
                softmax_ptr,
                target_ptr,
                weight_ptr,
                grad_input_ptr,
                gamma,
                alpha,
                softmax.size(1));
    } else {
        xav::cpu::softmax_focal_loss_grad(
                ctx,
                softmax.size(0),
                softmax_ptr,
                target_ptr,
                weight_ptr,
                buff_ptr,
                grad_input_ptr,
                gamma,
                alpha,
                softmax.size(1));
    }
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("softmax_focal_loss_forward", &softmax_focal_loss_forward);
    m.impl("softmax_focal_loss_backward", &softmax_focal_loss_backward);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("softmax_focal_loss_forward", &softmax_focal_loss_forward);
    m.impl("softmax_focal_loss_backward", &softmax_focal_loss_backward);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("softmax_focal_loss_forward(Tensor softmax, Tensor target, Tensor? weight, "
                                 "Tensor(a!) output, float gamma, float alpha) -> ()"));

    m.def(TORCH_SELECTIVE_SCHEMA("softmax_focal_loss_backward(Tensor softmax, Tensor target, Tensor? weight, "
                                 "Tensor(a!) buff, Tensor(b!) grad_input, float gamma, float alpha) -> ()"));
}
