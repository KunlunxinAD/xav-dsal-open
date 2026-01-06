#include "xpytorch.hpp"
#include <vector>
#include <iostream>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define MODE_BOUNDARY 0
#define MODE_MASK 1
#define MODE_HARD_MASK 2

std::vector<at::Tensor> forward_rasterize(
        at::Tensor vertices,
        at::Tensor rasterized,
        at::Tensor contribution_map,
        int64_t width,
        int64_t height,
        double inv_smoothness_,
        int64_t mode) {
    float inv_smoothness = static_cast<float>(inv_smoothness_);
    CHECK_CONTIGUOUS(vertices);
    CHECK_CONTIGUOUS(rasterized);
    CHECK_CONTIGUOUS(contribution_map);

    // return forward_rasterize_cuda(vertices, rasterized, contribution_map, width, height, inv_smoothness, mode);
    const auto batch_size = vertices.size(0);
    const auto number_vertices = vertices.size(1);
    auto ctx = xmlir_rt::getXpuKernelContext();

    auto kernel = xav::cpu::forward_rasterize<float>;
    if (vertices.device().is_cuda()) {
        kernel = xav::xpu::forward_rasterize<float>;
    }
    kernel(ctx,
           vertices.data_ptr<float>(),
           rasterized.data_ptr<float>(),
           contribution_map.data_ptr<int>(),
           batch_size,
           number_vertices,
           width,
           height,
           inv_smoothness,
           mode);

    return {rasterized, contribution_map};
}

at::Tensor backward_rasterize(
        at::Tensor vertices,
        at::Tensor rasterized,
        at::Tensor contribution_map,
        at::Tensor grad_output,
        at::Tensor grad_vertices,
        int64_t width,
        int64_t height,
        double inv_smoothness_,
        int64_t mode) {
    float inv_smoothness = static_cast<float>(inv_smoothness_);
    CHECK_CONTIGUOUS(vertices);
    CHECK_CONTIGUOUS(rasterized);
    CHECK_CONTIGUOUS(contribution_map);
    CHECK_CONTIGUOUS(grad_output);
    CHECK_CONTIGUOUS(grad_vertices);

    // return backward_rasterize_cuda(vertices, rasterized, contribution_map, grad_output, grad_vertices, width, height,
    // inv_smoothness, mode);
    const auto batch_size = vertices.size(0);
    const auto number_vertices = vertices.size(1);
    auto ctx = xmlir_rt::getXpuKernelContext();

    auto kernel = xav::cpu::backward_rasterize<float>;
    if (vertices.device().is_cuda()) {
        kernel = xav::xpu::backward_rasterize<float>;
    }
    kernel(ctx,
           vertices.data_ptr<float>(),
           rasterized.data_ptr<float>(),
           contribution_map.data_ptr<int>(),
           grad_output.data_ptr<float>(),
           grad_vertices.data_ptr<float>(),
           batch_size,
           number_vertices,
           width,
           height,
           inv_smoothness,
           mode);

    return grad_vertices;
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("forward_rasterize", &forward_rasterize);
    m.impl("backward_rasterize", &backward_rasterize);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("forward_rasterize", &forward_rasterize);
    m.impl("backward_rasterize", &backward_rasterize);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("forward_rasterize(Tensor vertices, Tensor rasterized, Tensor contribution_map, "
                                 "int width, int height, float inv_smoothness, int mode) -> Tensor[]"));

    m.def(TORCH_SELECTIVE_SCHEMA(
            "backward_rasterize(Tensor vertices, Tensor rasterized, Tensor contribution_map, "
            "Tensor grad_output, Tensor grad_vertices, int width, int height, float inv_smoothness, "
            "int mode) -> Tensor"));
}