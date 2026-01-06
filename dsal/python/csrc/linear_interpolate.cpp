#include "xpytorch.hpp"

at::Tensor linear_interpolate(at::Tensor batch_points, int64_t num_points) {
    auto ctx = xmlir_rt::getXpuKernelContext();
    int ret = 0;

    int batch_size = batch_points.sizes()[0];
    int n = batch_points.sizes()[1];
    at::Tensor output = at::empty({batch_size, num_points, 2}, batch_points.options());

    auto kernel = xav::cpu::linear_interpolate;
    if (batch_points.device().is_cuda())
        kernel = xav::xpu::linear_interpolate;
    kernel(ctx, batch_points.data_ptr<float>(), output.data_ptr<float>(), num_points, batch_size, n);
    assert(ret == 0);

    return output;
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("linear_interpolate", &linear_interpolate);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("linear_interpolate", &linear_interpolate);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("linear_interpolate(Tensor batch_points, int num_points) -> Tensor"));
}
