#include <torch/torch.h>
#include <vector>
#include <iostream>
#include "xpytorch.hpp"

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

std::vector<at::Tensor> forward_rasterize_xtrans_(
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

    const auto batch_size = vertices.size(0);
    const auto number_vertices = vertices.size(1);
    const int64_t threads = 512;

    auto ctx = xmlir_rt::getXpuKernelContext();
	auto kernel = xav::xpu::forward_rasterize_xtrans;
    int64_t result = kernel(ctx, threads,
                            vertices.data<float>(),
                            batch_size,
                            number_vertices,
                            rasterized.data<float>(),
                            contribution_map.data<int>(),
                            height,
                            width,
                            inv_smoothness,
                            mode);

    // AT_DISPATCH_FLOATING_TYPES(vertices.type(), "forward_rasterize_cuda", ([&] {
    //     forward_rasterize_cuda<float>(threads,
    //             vertices.data<float>(),
    //             batch_size,
    //             number_vertices,
    //             rasterized.data<float>(),
    //             height,
    //             width,
    //             inv_smoothness,
    //             mode);
    //     }));
    return {rasterized, contribution_map};
}

at::Tensor backward_rasterize_xtrans_(
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

    const auto batch_size = vertices.size(0);
    const auto number_vertices = vertices.size(1);
    const int64_t threads = 512;

    auto ctx = xmlir_rt::getXpuKernelContext();
	auto kernel = xav::xpu::backward_rasterize_xtrans;
    int64_t result = kernel(ctx, threads,
                            vertices.data<float>(),
                            rasterized.data<float>(),
                            contribution_map.data<int>(),
                            grad_output.data<float>(),
                            grad_vertices.data<float>(),
                            batch_size,
                            number_vertices,
                            width,
                            height,
                            inv_smoothness);

    // AT_DISPATCH_FLOATING_TYPES(vertices.type(), "backward_rasterize_cuda", ([&] {
    //     backward_rasterize_cuda<float>(threads,
    //             vertices.data<float>(),
    //             rasterized.data<float>(),
    //             contribution_map.data<int64_t>(),
    //             grad_output.data<float>(),
    //             grad_vertices.data<float>(),
    //             batch_size,
    //             number_vertices,
    //             width,
    //             height,
    //             inv_smoothness);
    //     }));

    return grad_vertices;
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("forward_rasterize_xtrans", &forward_rasterize_xtrans_);
    m.impl("backward_rasterize_xtrans", &backward_rasterize_xtrans_);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("forward_rasterize_xtrans", &forward_rasterize_xtrans_);
    m.impl("backward_rasterize_xtrans", &backward_rasterize_xtrans_);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("forward_rasterize_xtrans(Tensor vertices, Tensor rasterized, Tensor contribution_map, "
                                 "int width, int height, float inv_smoothness, int mode) -> Tensor[]"));

    m.def(TORCH_SELECTIVE_SCHEMA(
            "backward_rasterize_xtrans(Tensor vertices, Tensor rasterized, Tensor contribution_map, "
            "Tensor grad_output, Tensor grad_vertices, int width, int height, float inv_smoothness, "
            "int mode) -> Tensor"));
}
